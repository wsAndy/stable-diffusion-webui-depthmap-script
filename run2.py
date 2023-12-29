
import os
import traceback
from pathlib import Path
import pathlib

try:
    from tqdm import trange 
except:
    from builtins import range as trange

from PIL import Image
import cv2
import uuid
from src import video_mode
# from src.core import run_makevideo
from src.depthmap_generation import ModelHolder
from src.stereoimage_generation import create_stereoimages
from src.normalmap_generation import create_normalmap
from src.misc import get_commit_hash
import numpy as np
import torch
import gc

from loguru import logger
import yaml
import datetime 
import time

def format_exception(e: Exception):
    traceback.print_exc()
    msg = '<h3>' + 'ERROR: ' + str(e) + '</h3>' + '\n'
    if 'out of GPU memory' not in msg:
        msg += 'Please report this issue ' + traceback.format_exc()
    return msg


inputs = {}

inputs['depthmap_script_keepmodels'] = False
inputs['output_path'] = "/code/outputs/1229" #"/code/outputs/img_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
inputs['compute_device'] = 'GPU'

inputs['model_type'] = 9    # zoedepth_nk

[w, h] = ModelHolder.get_default_net_size(inputs['model_type'])
inputs['net_width'] = w
inputs['net_height'] = h

inputs['net_size_match'] = False

inputs['save_outputs'] = True

inputs['stereo_balance'] = 0
inputs['stereo_divergence'] = 2.5
inputs['stereo_fill_algo'] = 'polylines_sharp'
inputs['stereo_modes'] = ['left-right']
inputs['stereo_offset_exponent'] = 2
inputs['stereo_separation'] = 0
inputs['do_output_depth'] = False


inputs['boost'] = False
inputs['clipdepth'] = False
inputs['clipdepth_far'] = 0
inputs['clipdepth_mode'] = 'Range'
inputs['clipdepth_near'] = 1


inputs['depthmap_vm_compress_bitrate'] = 4000 # 720p的差不多了
inputs['depthmap_vm_compress_checkbox'] = True # avi to mp4
inputs['depthmap_vm_smoothening_mode'] = 'experimental'
inputs['depthmap_vm_custom'] = None
inputs['depthmap_vm_custom_checkbox'] = False
inputs['depthmap_vm_input'] = None

inputs['gen_inpainted_mesh'] = False
inputs['gen_inpainted_mesh_demos'] = False
inputs['gen_normalmap'] = False
# inputs['gen_simple_mesh'] = False
inputs['gen_stereo'] = True
inputs['image_batch'] = None

inputs['normalmap_invert'] = False
inputs['normalmap_post_blur'] = False
inputs['normalmap_post_blur_kernel'] = 3
inputs['normalmap_pre_blur'] = False
inputs['normalmap_pre_blur_kernel'] = 3
inputs['normalmap_sobel'] = True
inputs['normalmap_sobel_kernel'] = 3

inputs['output_depth_combine'] = False # 用于将输入rgb和depth合并在一块
inputs['output_depth_combine_axis'] = 'Horizontal'
inputs['output_depth_invert'] = False
inputs['pre_depth_background_removal'] = False

inputs['rembg_model'] = 'u2net'
inputs['gen_rembg'] = False # remove background
inputs['save_background_removal_masks'] = False

inputs['simple_mesh_occlude'] = False
inputs['simple_mesh_spherical'] = False


######## load model


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

model_holder = ModelHolder()

def InitModel():

    model_holder.update_settings(None)

    # init torch device
    if inputs['compute_device'] == 'GPU':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print('WARNING: Cuda device was not found, cpu will be used')
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("device: %s" % device)

    try:
        print("Loading model(s) ..")
        model_holder.ensure_models(inputs['model_type'], device, inputs['boost'] )
        print("Computing output(s) ..")
    except Exception as e:
        import traceback
        if 'out of memory' in str(e).lower():
            print(str(e))
            suggestion = "out of GPU memory, could not generate depthmap! " \
                            "Here are some suggestions to work around this issue:\n"
            if device != torch.device("cpu"):
                suggestion += " * Select CPU as the processing device (this will be slower)\n"
            if inputs['model_type'] != 6:
                suggestion +=\
                    " * Use a different model (generally, more memory-consuming models produce better depthmaps)\n"
            print('Fail.\n')
            raise Exception(suggestion)
        else:
            print('Fail.\n')
            if inputs['depthmap_script_keepmodels'] == True:
                model_holder.offload()  # Swap to CPU memory
            else:
                model_holder.unload_models()
            gc.collect()
            torch_gc()
            raise e
    ######################


def batched_background_removal(inimages, model_name):
    from rembg import new_session, remove
    print('creating background masks')
    outimages = []

    # model path and name
    bg_model_dir = Path.joinpath(Path().resolve(), "models/rem_bg")
    os.makedirs(bg_model_dir, exist_ok=True)
    os.environ["U2NET_HOME"] = str(bg_model_dir)

    # starting a session
    background_removal_session = new_session(model_name)
    for count in range(0, len(inimages)):
        bg_remove_img = np.array(remove(inimages[count], session=background_removal_session))
        outimages.append(Image.fromarray(bg_remove_img))
    # The line below might be redundant
    del background_removal_session
    return outimages

def convert_to_i16(arr):
    # Single channel, 16 bit image. This loses some precision!
    # uint16 conversion uses round-down, therefore values should be [0; 2**16)
    numbytes = 2
    max_val = (2 ** (8 * numbytes))
    out = np.clip(arr * max_val + 0.0001, 0, max_val - 0.1)  # -0.1 from above is needed to avoid overflowing
    return out.astype("uint16")

def convert_i16_to_rgb(image, like):
    # three channel, 8 bits per channel image
    output = np.zeros_like(like)
    output[:, :, 0] = image / 256.0
    output[:, :, 1] = image / 256.0
    output[:, :, 2] = image / 256.0
    return output

def get_time_uuid():
    basecount = str( datetime.datetime.utcnow() ).replace('-','').replace(' ', '-').replace('.','-').replace(':','') #+ str(uuid.uuid4())[:6]
    return f"{basecount}-{get_commit_hash()}"

def get_uniquefn(outpath, basename, ext):
    os.makedirs(outpath, exist_ok=True)
    return os.path.join(outpath, f"{basename}-{get_time_uuid()}.{ext}")


def GetMonoDepth( input_images_path = [], input_depth_path = [], ops = {} ):

    if 'gen_normalmap' not in ops:
        genNormalMap = inputs['gen_normalmap']
    else:
        genNormalMap = ops['gen_normalmap']

    if 'gen_stereo' not in ops:
        genStereo = inputs['gen_stereo']
    else:
        genStereo = ops['gen_stereo']

    if 'do_output_depth' not in ops:
        genOutputDepth = inputs['do_output_depth']
    else:
        genOutputDepth = ops['do_output_depth']

    if 'gen_rembg' not in ops:
        genRembg = inputs['gen_rembg']
    else:
        genRembg = ops['gen_rembg']
    
    inputimages = []
    inputdepthmaps = []  # Allow supplying custom depthmaps
    inputnames = []  # Also keep track of original file names

    hasCustomDepth = False

    if len(input_depth_path) != 0:
        hasCustomDepth = True
        if len(input_depth_path) != len(input_images_path):
            logger.error('custom depth array size need equals to input image array size')
            return

    for index in range(len(input_images_path)):
        img = input_images_path[index]
        if isinstance(img, np.ndarray):
            inputimages.append( Image.fromarray(img) )
        elif isinstance(img, Image.Image):
            inputimages.append( img )
        elif isinstance(img, str):
            if os.path.exists(img) == False:
                continue
            inputimages.append( Image.open(img) )

        if hasCustomDepth:
            dep = input_depth_path[index]
            if isinstance(dep, np.ndarray):
                inputdepthmaps.append( Image.fromarray(dep) )
            elif isinstance(dep, str):
                if os.path.exists(dep) == False:
                    continue
                else:
                    inputdepthmaps.append( PIL.open(dep) )
            elif isinstance(dep, Image.Image):
                inputdepthmaps.append( dep )
        else:
            inputdepthmaps.append(None) # 自定义的depth
        inputnames.append(None)

    if len(inputimages) == 0:
        logger.error('custom depth array size need equals to input image array size')
        return
    if hasCustomDepth:
        if len(inputimages) != len(inputdepthmaps):
            logger.error('custom depth array need string or PIL.Image or np.ndarray')
            return
    
    inputdepthmaps_complete = all([x is not None for x in inputdepthmaps])

    background_removed_images = []
    if inputs['gen_rembg']:
        if inputs['pre_depth_background_removal']:
            inputimages = batched_background_removal(inputimages, inputs['rembg_model'] )
            background_removed_images = inputimages
        else:
            background_removed_images = batched_background_removal(inputimages, inputs['rembg_model'] )

    inpaint_imgs = []
    inpaint_depths = []

    try:
        for count in trange(0, len(inputimages)):
            # Convert single channel input (PIL) images to rgb
            if inputimages[count].mode == 'I':
                inputimages[count].point(lambda p: p * 0.0039063096, mode='RGB')
                inputimages[count] = inputimages[count].convert('RGB')
            if inputimages[count].mode == 'RGBA':
                inputimages[count] = inputimages[count].convert('RGB')

            raw_prediction = None
            """Raw prediction, as returned by a model. None if input depthmap is used."""
            raw_prediction_invert = False
            """True if near=dark on raw_prediction"""
            out = None

            if inputdepthmaps is not None and inputdepthmaps[count] is not None:
                # use custom depthmap
                dp = inputdepthmaps[count]
                if isinstance(dp, Image.Image):
                    if dp.width != inputimages[count].width or dp.height != inputimages[count].height:
                        try:  # LANCZOS may fail on some formats
                            dp = dp.resize((inputimages[count].width, inputimages[count].height), Image.Resampling.LANCZOS)
                        except:
                            dp = dp.resize((inputimages[count].width, inputimages[count].height))
                    # Trying desperately to rescale image to [0;1) without actually normalizing it
                    # Normalizing is avoided, because we want to preserve the scale of the original depthmaps
                    # (batch mode, video mode).
                    if len(dp.getbands()) == 1:
                        out = np.asarray(dp, dtype="float")
                        out_max = out.max()
                        if out_max < 256:
                            bit_depth = 8
                        elif out_max < 65536:
                            bit_depth = 16
                        else:
                            bit_depth = 32
                        out /= 2.0 ** bit_depth
                    else:
                        out = np.asarray(dp, dtype="float")[:, :, 0]
                        out /= 256.0
                else:
                    # Should be in interval [0; 1], values outside of this range will be clipped.
                    out = np.asarray(dp, dtype="float")
                    assert inputimages[count].height == out.shape[0], "Custom depthmap height mismatch"
                    assert inputimages[count].width == out.shape[1], "Custom depthmap width mismatch"
            else:
                # override net size (size may be different for different images)
                if inputs['net_size_match']:
                    # Round up to a multiple of 32 to avoid potential issues
                    net_width = (inputimages[count].width + 31) // 32 * 32
                    net_height = (inputimages[count].height + 31) // 32 * 32
                else:
                    net_width = inputs['net_width']
                    net_height = inputs['net_height']
                raw_prediction, raw_prediction_invert = model_holder.get_raw_prediction(inputimages[count], net_width, net_height)

                # output
                if abs(raw_prediction.max() - raw_prediction.min()) > np.finfo("float").eps:
                    out = np.copy(raw_prediction)
                    # TODO: some models may output negative values, maybe these should be clamped to zero.
                    if raw_prediction_invert:
                        out *= -1
                    if inputs['clipdepth']:
                        if inputs['clipdepth_mode'] == 'Range':
                            out = (out - out.min()) / (out.max() - out.min())  # normalize to [0; 1]
                            out = np.clip(out, inputs['clipdepth_far'], inputs['clipdepth_near'])
                        elif inputs['clipdepth_mode'] == 'Outliers':
                            fb, nb = np.percentile(out, [inputs['clipdepth_far'] * 100.0, inputs['clipdepth_near'] * 100.0])
                            out = np.clip(out, fb, nb)
                    out = (out - out.min()) / (out.max() - out.min())  # normalize to [0; 1]
                else:
                    # Regretfully, the depthmap is broken and will be replaced with a black image
                    out = np.zeros(raw_prediction.shape)

            # Maybe we should not use img_output for everything, since we get better accuracy from
            # the raw_prediction. However, it is not always supported. We maybe would like to achieve
            # reproducibility, so depthmap of the image should be the same as generating the depthmap one more time.
            img_output = convert_to_i16(out)
            """Depthmap (near=bright), as uint16"""

            # if 3dinpainting, store maps for processing in second pass
            if inputs['gen_inpainted_mesh']:
                inpaint_imgs.append(inputimages[count])
                inpaint_depths.append(img_output)

            # applying background masks after depth
            if genRembg:
                print('applying background masks')
                background_removed_image = background_removed_images[count]
                # maybe a threshold cut would be better on the line below.
                background_removed_array = np.array(background_removed_image)
                bg_mask = (background_removed_array[:, :, 0] == 0) & (background_removed_array[:, :, 1] == 0) & (
                        background_removed_array[:, :, 2] == 0) & (background_removed_array[:, :, 3] <= 0.2)
                img_output[bg_mask] = 0  # far value

                yield count, 'background_removed', background_removed_image

                if inputs['save_background_removal_masks']:
                    bg_array = (1 - bg_mask.astype('int8')) * 255
                    mask_array = np.stack((bg_array, bg_array, bg_array, bg_array), axis=2)
                    mask_image = Image.fromarray(mask_array.astype(np.uint8))

                    yield count, 'foreground_mask', mask_image

            # A weird quirk: if user tries to save depthmap, whereas custom depthmap is used,
            # custom depthmap will be outputed
            if genOutputDepth:
                img_depth = cv2.bitwise_not(img_output) if inputs['output_depth_invert'] else img_output
                if inputs['output_depth_combine']:
                    axis = 1 if inputs['output_depth_combine_axis'] == 'Horizontal' else 0
                    img_concat = Image.fromarray(np.concatenate(
                        (inputimages[count], convert_i16_to_rgb(img_depth, inputimages[count])),
                        axis=axis))
                    yield count, 'concat_depth', img_concat
                else:
                    yield count, 'depth', Image.fromarray(img_depth)

            if genStereo:
                # print("Generating stereoscopic image(s)..")
                stereoimages = create_stereoimages(
                    inputimages[count], img_output,
                    inputs['stereo_divergence'], inputs['stereo_separation'],
                    inputs['stereo_modes'],
                    inputs['stereo_balance'], inputs['stereo_offset_exponent'], inputs['stereo_fill_algo'])
                for c in range(0, len(stereoimages)):
                    yield count, inputs['stereo_modes'][c], stereoimages[c]

            if genNormalMap:
                normalmap = create_normalmap(
                    img_output,
                    inputs['normalmap_pre_blur_kernel'] if inputs['normalmap_pre_blur'] else None,
                    inputs['normalmap_sobel_kernel'] if inputs['normalmap_sobel'] else None,
                    inputs['normalmap_post_blur_kernel'] if inputs['normalmap_post_blur'] else None,
                    inputs['normalmap_invert']
                )
                yield count, 'normalmap', normalmap

            # # gen mesh
            # if inputs['gen_simple_mesh']:
            #     print(f"\nGenerating (occluded) mesh ..")
            #     basename = 'depthmap'
            #     meshsimple_fi = get_uniquefn( inputs['output_path'], basename, 'obj')

            #     depthi = raw_prediction if raw_prediction is not None else out
            #     depthi_min, depthi_max = depthi.min(), depthi.max()
            #     # try to map output to sensible values for non zoedepth models, boost, or custom maps
            #     if inputs['model_type'] not in [7, 8, 9] or inputs['boost'] or inputdepthmaps[count] is not None:
            #         # invert if midas
            #         if inputs['model_type'] > 0 or inputdepthmaps[count] is not None:  # TODO: Weird
            #             depthi = depthi_max - depthi + depthi_min
            #             depth_max = depthi.max()
            #             depth_min = depthi.min()
            #         # make positive
            #         if depthi_min < 0:
            #             depthi = depthi - depthi_min
            #             depth_max = depthi.max()
            #             depth_min = depthi.min()
            #         # scale down
            #         if depthi.max() > 10.0:
            #             depthi = 4.0 * (depthi - depthi_min) / (depthi_max - depthi_min)
            #         # offset
            #         depthi = depthi + 1.0

            #     mesh = create_mesh(inputimages[count], depthi, keep_edges=not inputs['simple_mesh_occlude'],
            #                        spherical=(inputs['simple_mesh_spherical']))
            #     mesh.export(meshsimple_fi)
            #     yield count, 'simple_mesh', meshsimple_fi

        print("Computing output(s) done.")
    except Exception as e:
        import traceback
        if 'out of memory' in str(e).lower():
            print(str(e))
            suggestion = "out of GPU memory, could not generate depthmap! " \
                            "Here are some suggestions to work around this issue:\n"
            if device != torch.device("cpu"):
                suggestion += " * Select CPU as the processing device (this will be slower)\n"
            if inputs['model_type'] != 6:
                suggestion +=\
                    " * Use a different model (generally, more memory-consuming models produce better depthmaps)\n"
            print('Fail.\n')
            raise Exception(suggestion)
        else:
            print('Fail.\n')
            raise e
    
def doGC():
    gc.collect()
    torch_gc()

def finishInference():
    if inputs['depthmap_script_keepmodels'] == True:
        model_holder.offload()  # Swap to CPU memory
    else:
        model_holder.unload_models()
    doGC()



def open_path_as_images(path, maybe_depthvideo=False):
    """Takes the filepath, returns (fps, frames). Every frame is a Pillow Image object"""
    suffix = pathlib.Path(path).suffix
    if suffix.lower() == '.gif':
        frames = []
        img = Image.open(path)
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert('RGB'))
        return 1000 / img.info['duration'], frames, None
    if suffix.lower() == '.mts':
        import imageio_ffmpeg
        import av
        container = av.open(path)
        frames = []
        for packet in container.demux(video=0):
            for frame in packet.decode():
                # Convert the frame to a NumPy array
                numpy_frame = frame.to_ndarray(format='rgb24')
                # Convert the NumPy array to a Pillow Image
                image = Image.fromarray(numpy_frame)
                frames.append(image)
        fps = float(container.streams.video[0].average_rate)
        container.close()
        return fps, frames, None
    if suffix.lower() in ['.avi'] and maybe_depthvideo:
        try:
            import imageio_ffmpeg
            # Suppose there are in fact 16 bits per pixel
            # If this is not the case, this is not a 16-bit depthvideo, so no need to process it this way
            gen = imageio_ffmpeg.read_frames(path, pix_fmt='gray16le', bits_per_pixel=16)
            video_info = next(gen)
            if video_info['pix_fmt'] == 'gray16le':
                width, height = video_info['size']
                frames = []
                for frame in gen:
                    # Not sure if this is implemented somewhere else
                    result = np.frombuffer(frame, dtype='uint16')
                    result.shape = (height, width)  # Why does it work? I don't remotely have any idea.
                    frames += [Image.fromarray(result)]
                    # TODO: Wrapping frames into Pillow objects is wasteful
                return video_info['fps'], frames, None
        finally:
            if 'gen' in locals():
                gen.close()
    if suffix.lower() in ['.webm', '.mp4', '.avi']:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        frames = [Image.fromarray(x) for x in list(clip.iter_frames())]
        # TODO: Wrapping frames into Pillow objects is wasteful
        return clip.fps, frames, clip.audio
    else:
        try:
            return 1, [Image.open(path)]
        except Exception as e:
            raise Exception(f"Probably an unsupported file format: {suffix}") from e


def frames_to_video(fps, frames, sound, path, name, colorvids_bitrate=None):
    if frames[0].mode == 'I;16':  # depthmap video
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            os.path.join(path, f"{name}.avi"), frames[0].size, 'gray16le', 'gray16le', fps, codec='ffv1',
            macro_block_size=1)
        try:
            writer.send(None)
            for frame in frames:
                writer.send(np.array(frame))
        finally:
            writer.close()
    else:
        arrs = [np.asarray(frame) for frame in frames]
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(arrs, fps=fps)
        done = False
        priority = [('avi', 'png'), ('avi', 'rawvideo'), ('mp4', 'libx264') ] #, ('webm', 'libvpx')]
        if colorvids_bitrate:
            priority = reversed(priority)
        for v_format, codec in priority:
            try:
                br = f'{colorvids_bitrate}k' if codec not in ['png', 'rawvideo'] else None
                clip.audio = sound
                clip.write_videofile(os.path.join(path, f"{name}.{v_format}"), codec=codec, bitrate=br, audio_codec="aac")
                done = True
                break
            except:
                traceback.print_exc()
        if not done:
            raise Exception('Saving the video failed!')


def process_predicitons(predictions, smoothening='none'):
    def global_scaling(objs, a=None, b=None):
        """Normalizes objs, but uses (a, b) instead of (minimum, maximum) value of objs, if supplied"""
        normalized = []
        min_value = a if a is not None else min([obj.min() for obj in objs])
        max_value = b if b is not None else max([obj.max() for obj in objs])
        for obj in objs:
            normalized += [(obj - min_value) / (max_value - min_value)]
        return normalized
    
    predictions = [ np.asarray(x).astype(float) for x in predictions if isinstance(x, Image.Image)]
    print('Processing generated depthmaps')
    # TODO: Detect cuts and process segments separately
    if smoothening == 'none':
        predictions = global_scaling(predictions)
    elif smoothening == 'experimental':
        processed = []
        clip = lambda val: min(max(0, val), len(predictions) - 1)
        for i in range(len(predictions)):
            f = np.zeros_like(predictions[i])
            for u, mul in enumerate([0.10, 0.20, 0.40, 0.20, 0.10]):  # Eyeballed it, math person please fix this
                f += mul * predictions[clip(i + (u - 2))]
            processed += [f]
        # This could have been deterministic monte carlo... Oh well, this version is faster.
        a, b = np.percentile(np.stack(processed), [0.5, 99.5])
        predictions = global_scaling(predictions, a, b)
    predictions = [ Image.fromarray(x) for x in predictions ]
    return predictions


def gen_video(videos, custom_depthmaps, outpath, colorvids_bitrate=None, smoothening='none'):
    # if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
    #     return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    for index in range(len(videos)):
        startTime = time.time()

        video = videos[index]
        if custom_depthmaps is None:
            custom_depthmap = None
        else:
            custom_depthmap = custom_depthmaps[index]

        fps, input_images, sound = open_path_as_images(os.path.abspath(video))
        os.makedirs(inputs['output_path'], exist_ok=True)

        if custom_depthmap is None:
            print('Generating depthmaps for the video frames')
            # firstly, create depth only
            gen_obj = GetMonoDepth( input_images, [], {'gen_normalmap': False, 'gen_stereo': False, 'do_output_depth': True, 'gen_rembg': False} )
            input_depths = [x[2] for x in list(gen_obj)]
            # 全部是PIL.Image.Image 类型
            input_depths = process_predicitons(input_depths, smoothening)
        else:
            print('Using custom depthmap video')
            cdm_fps, input_depths, _ = open_path_as_images(os.path.abspath(custom_depthmap), maybe_depthvideo=True)
            assert len(input_depths) == len(input_images), 'Custom depthmap video length does not match input video length'
            if input_depths[0].size != input_images[0].size:
                print('Warning! Input video size and depthmap video size are not the same!')

        print('Generating output frames')
        img_results = list(GetMonoDepth( input_images, input_depths))
        gens = list(set(map(lambda x: x[1], img_results)))

        print('Saving generated frames as video outputs')
        for gen in gens:
            imgs = [x[2] for x in img_results if x[1] == gen]
            
            basename = f'{os.path.basename(video).split(".")[0]}_{gen}_video'
            frames_to_video(fps, imgs, sound, outpath, f"{basename}-{get_time_uuid()}", colorvids_bitrate)

        logger.info("{0} cost time: {1:.2f}, size: {2}x{3}, number: {4}".format(video, time.time()-startTime, input_images[0].size[0], input_images[0].size[1], len(input_images) ) )
        doGC()
    print('All done. Video(s) saved!')



if __name__ == '__main__':
    logger.add("2dto3d.log")

    InitModel()
    data = []
    # data = [ os.path.join("/code/data/img/", x ) for x in os.listdir("/code/data/img/") if x.lower().endswith('png')]
    # data = ["/code/data/img/0009.png"] #, "/code/data/img/0013.png", "/code/data/img/0014.png", "/code/data/img/0015.png", "/code/data/img/0016.png", "/code/data/img/0017.png"]
    
    dataVideo = ['/code/data/test.mp4'] # [ os.path.join("/code/data/1229/", x ) for x in os.listdir("/code/data/1229/") if x.lower().endswith('mp4')]
    # dataVideo = ['/code/data/v/env2.mp4', '/code/data/v/env3.mp4',  '/code/data/v/multi_people_dance.mp4',  '/code/data/v/solo_boy_dance1.mp4', '/code/data/v/solo_girl_dance2.mp4',  '/code/data/v/solo_girl_dance3.mp4',  '/code/data/v/solo_girl_dance4.mp4']
    # dataVideo = ["/code/data/dance1.mp4", "/code/data/dance2.mp4"]

    if len(data) > 0:
        gen_proc = GetMonoDepth(data)
        try:
            while True:
                input_i, type, result = next(gen_proc)

                if type == 'depth':
                    C = np.asarray(result)
                    Cmin = C.min()
                    Cmax = C.max()
                    c = (C - Cmin)/(Cmax - Cmin)
                    c = (c * 255).astype(np.uint8)
                    mask = c >= (np.mean(c) + np.max(c))/2
                    d = c * mask
                    dp = Image.fromarray(d)
                    dp.save('/code/data/img_nobg/{0}_d.png'.format( input_i ) )
                else:
                    continue

                outputFilePath = get_uniquefn(inputs['output_path'], type, 'png' )
                logger.info("images, {0}, {1}".format(outputFilePath, inputs) )
                result.save(outputFilePath )

        except StopIteration:
            print('===Down===')
        
    if len(dataVideo)>0:
        custom_depthmap = inputs['depthmap_vm_custom'] \
            if inputs['depthmap_vm_custom_checkbox'] else None
        colorvids_bitrate = inputs['depthmap_vm_compress_bitrate'] \
            if inputs['depthmap_vm_compress_checkbox'] else None
        gen_proc = gen_video(dataVideo, custom_depthmap, inputs['output_path'],  colorvids_bitrate, inputs['depthmap_vm_smoothening_mode'])

    # img_results = []

    # img_results += [(input_i, type, result)]

    # print(img_results)





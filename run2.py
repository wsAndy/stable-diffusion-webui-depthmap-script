
import os
import traceback
from pathlib import Path

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

def format_exception(e: Exception):
    traceback.print_exc()
    msg = '<h3>' + 'ERROR: ' + str(e) + '</h3>' + '\n'
    if 'out of GPU memory' not in msg:
        msg += 'Please report this issue ' + traceback.format_exc()
    return msg


inputs = {}

inputs['depthmap_script_keepmodels'] = True
inputs['output_path'] = "/code/outputs"
inputs['compute_device'] = 'GPU'
inputs['depthmap_input_image'] = Image.open('/code/data/0001.png') # PIL的一张原图
inputs['depthmap_mode'] = '0'  # 0: Single image, 1: Batch Process, 2: Batch Process From Directory, 3: video mode
inputs['model_type'] = 9    # zoedepth_nk

inputs['net_height'] = 512
inputs['net_size_match'] = False
inputs['net_width'] = 384

inputs['save_outputs'] = True

inputs['stereo_balance'] = -1
inputs['stereo_divergence'] = 2.5
inputs['stereo_fill_algo'] = 'polylines_sharp'
inputs['stereo_modes'] = ['left-right']
inputs['stereo_offset_exponent'] = 2
inputs['stereo_separation'] = 0

inputs['do_output_depth'] = True



inputs['boost'] = False
inputs['clipdepth'] = False
inputs['clipdepth_far'] = 0
inputs['clipdepth_mode'] = 'Range'
inputs['clipdepth_near'] = 1

inputs['depthmap_batch_input_dir'] = ''
inputs['depthmap_batch_output_dir'] = ''
inputs['depthmap_batch_reuse'] = True

inputs['depthmap_vm_compress_bitrate'] = 15000
inputs['depthmap_vm_compress_checkbox'] = False
inputs['depthmap_vm_custom'] = None
inputs['depthmap_vm_custom_checkbox'] = False
inputs['depthmap_vm_input'] = None
inputs['depthmap_vm_smoothening_mode'] = 'experimental'

inputs['gen_inpainted_mesh'] = False
inputs['gen_inpainted_mesh_demos'] = False
inputs['gen_normalmap'] = False
inputs['gen_simple_mesh'] = False
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
inputs['gen_rembg'] = False
inputs['save_background_removal_masks'] = False

inputs['simple_mesh_occlude'] = False
inputs['simple_mesh_spherical'] = False


inputs['custom_depthmap'] = False
inputs['custom_depthmap_img'] = None

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

def get_uniquefn(outpath, basename, ext, suffix=''):
    basecount = str( datetime.datetime.utcnow() ).replace('-','').replace(' ', '-').replace('.','-').replace(':','') #+ str(uuid.uuid4())[:6]
    os.makedirs(outpath, exist_ok=True)
    return os.path.join(outpath, f"{basename}-{basecount}-{get_commit_hash()}-{suffix}.{ext}")


def GetMonoDepth():
    
    inputimages = []
    inputdepthmaps = []  # Allow supplying custom depthmaps
    inputnames = []  # Also keep track of original file names

    inputimages.append(inputs['depthmap_input_image'])
    inputdepthmaps.append(None) # 自定义的depth
    inputnames.append(None)


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
            if inputs['gen_rembg']:
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
            if inputs['do_output_depth']:
                img_depth = cv2.bitwise_not(img_output) if inputs['output_depth_invert'] else img_output
                if inputs['output_depth_combine']:
                    axis = 1 if inputs['output_depth_combine_axis'] == 'Horizontal' else 0
                    img_concat = Image.fromarray(np.concatenate(
                        (inputimages[count], convert_i16_to_rgb(img_depth, inputimages[count])),
                        axis=axis))
                    yield count, 'concat_depth', img_concat
                else:
                    yield count, 'depth', Image.fromarray(img_depth)

            if inputs['gen_stereo']:
                # print("Generating stereoscopic image(s)..")
                stereoimages = create_stereoimages(
                    inputimages[count], img_output,
                    inputs['stereo_divergence'], inputs['stereo_separation'],
                    inputs['stereo_modes'],
                    inputs['stereo_balance'], inputs['stereo_offset_exponent'], inputs['stereo_fill_algo'])
                for c in range(0, len(stereoimages)):
                    yield count, inputs['stereo_modes'][c], stereoimages[c]

            if inputs['gen_normalmap']:
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
            #     meshsimple_fi = get_uniquefn( inputs['output_path'], basename, 'obj', 'simple')

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
    finally:
        if inputs['depthmap_script_keepmodels'] == True:
            model_holder.offload()  # Swap to CPU memory
        else:
            model_holder.unload_models()
        gc.collect()
        torch_gc()


if __name__ == '__main__':
    logger.add("2dto3d.log")

    InitModel()
    gen_proc = GetMonoDepth()
    
    img_results = []

    try:
        while True:
            input_i, type, result = next(gen_proc)
            outputFilePath = get_uniquefn(inputs['output_path'], type, 'png', '')
            logger.info("images, {0}, {1}".format(outputFilePath, inputs) )
            result.save(outputFilePath )

    except StopIteration:
        print('===Down===')
    
    # img_results += [(input_i, type, result)]

    # print(img_results)

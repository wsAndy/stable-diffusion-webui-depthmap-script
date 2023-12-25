
import os
import traceback
from pathlib import Path

from PIL import Image

from src import backbone, video_mode
from src.core import core_generation_funnel, unload_models, run_makevideo
from src.depthmap_generation import ModelHolder
from src.gradio_args_transport import GradioComponentBundle
from src.misc import *
from src.common_constants import GenerationOptions as go


def format_exception(e: Exception):
    traceback.print_exc()
    msg = '<h3>' + 'ERROR: ' + str(e) + '</h3>' + '\n'
    if 'out of GPU memory' not in msg:
        msg += 'Please report this issue ' + traceback.format_exc()
    return msg


inputs = {}
inputs['boost'] = False
inputs['clipdepth'] = False
inputs['clipdepth_far'] = 0
inputs['clipdepth_mode'] = 'Range'
inputs['clipdepth_near'] = 1

inputs['compute_device'] = 'GPU'

inputs['custom_depthmap'] = False
inputs['custom_depthmap_img'] = None

inputs['depthmap_batch_input_dir'] = ''
inputs['depthmap_batch_output_dir'] = ''
inputs['depthmap_batch_reuse'] = True
inputs['depthmap_input_image'] = Image.open('/code/data/0001.png') # PIL的一张原图
inputs['depthmap_mode'] = '0'  # 0: Single image, 1: Batch Process, 2: Batch Process From Directory, 3: video mode

inputs['depthmap_vm_compress_bitrate'] = 15000
inputs['depthmap_vm_compress_checkbox'] = False
inputs['depthmap_vm_custom'] = None
inputs['depthmap_vm_custom_checkbox'] = False
inputs['depthmap_vm_input'] = None
inputs['depthmap_vm_smoothening_mode'] = 'experimental'

inputs['do_output_depth'] = False
inputs['gen_inpainted_mesh'] = False
inputs['gen_inpainted_mesh_demos'] = False
inputs['gen_normalmap'] = False
inputs['gen_rembg'] = False
inputs['gen_simple_mesh'] = False
inputs['gen_stereo'] = True
inputs['image_batch'] = None
inputs['model_type'] = 9    # zoedepth_nk

inputs['net_height'] = 512
inputs['net_size_match'] = False
inputs['net_width'] = 384

inputs['normalmap_invert'] = False
inputs['normalmap_post_blur'] = False
inputs['normalmap_post_blur_kernel'] = 3
inputs['normalmap_pre_blur'] = False
inputs['normalmap_pre_blur_kernel'] = 3
inputs['normalmap_sobel'] = True
inputs['normalmap_sobel_kernel'] = 3

inputs['output_depth_combine'] = False
inputs['output_depth_combine_axis'] = 'Horizontal'
inputs['output_depth_invert'] = False
inputs['pre_depth_background_removal'] = False
inputs['rembg_model'] = 'u2net'
inputs['save_background_removal_masks'] = False
inputs['save_outputs'] = True
inputs['simple_mesh_occlude'] = False
inputs['simple_mesh_spherical'] = False

inputs['stereo_balance'] = -1
inputs['stereo_divergence'] = 2.5
inputs['stereo_fill_algo'] = 'polylines_sharp'
inputs['stereo_modes'] = ['left-right']
inputs['stereo_offset_exponent'] = 2
inputs['stereo_separation'] = 0


def run_generate(inputs):
    depthmap_mode = inputs['depthmap_mode']
    depthmap_batch_input_dir = inputs['depthmap_batch_input_dir']
    image_batch = inputs['image_batch']
    depthmap_input_image = inputs['depthmap_input_image']
    depthmap_batch_output_dir = inputs['depthmap_batch_output_dir']
    depthmap_batch_reuse = inputs['depthmap_batch_reuse']
    custom_depthmap = inputs['custom_depthmap']
    custom_depthmap_img = inputs['custom_depthmap_img']

    inputimages = []
    inputdepthmaps = []  # Allow supplying custom depthmaps
    inputnames = []  # Also keep track of original file names

    if depthmap_mode == '3':
        try:
            custom_depthmap = inputs['depthmap_vm_custom'] \
                if inputs['depthmap_vm_custom_checkbox'] else None
            colorvids_bitrate = inputs['depthmap_vm_compress_bitrate'] \
                if inputs['depthmap_vm_compress_checkbox'] else None
            ret = video_mode.gen_video(
                inputs['depthmap_vm_input'], backbone.get_outpath(), inputs, custom_depthmap, colorvids_bitrate,
                inputs['depthmap_vm_smoothening_mode'])
            return [], None, None, ret
        except Exception as e:
            ret = format_exception(e)
        return [], None, None, ret

    if depthmap_mode == '2' and depthmap_batch_output_dir != '':
        outpath = depthmap_batch_output_dir
    else:
        outpath = backbone.get_outpath()

    if depthmap_mode == '0':  # Single image
        if depthmap_input_image is None:
            return [], None, None, "Please select an input image"
        inputimages.append(depthmap_input_image)
        inputnames.append(None)
        if custom_depthmap:
            if custom_depthmap_img is None:
                return [], None, None, \
                    "Custom depthmap is not specified. Please either supply it or disable this option."
            inputdepthmaps.append(Image.open(os.path.abspath(custom_depthmap_img.name)))
        else:
            inputdepthmaps.append(None)
    if depthmap_mode == '1':  # Batch Process
        if image_batch is None:
            return [], None, None, "Please select input images", ""
        for img in image_batch:
            image = Image.open(os.path.abspath(img.name))
            inputimages.append(image)
            inputnames.append(os.path.splitext(img.orig_name)[0])
        print(f'{len(inputimages)} images will be processed')
    elif depthmap_mode == '2':  # Batch from Directory
        # TODO: There is a RAM leak when we process batches, I can smell it! Or maybe it is gone.
        assert not backbone.get_cmd_opt('hide_ui_dir_config', False), '--hide-ui-dir-config option must be disabled'
        if depthmap_batch_input_dir == '':
            return [], None, None, "Please select an input directory."
        if depthmap_batch_input_dir == depthmap_batch_output_dir:
            return [], None, None, "Please pick different directories for batch processing."
        image_list = backbone.listfiles(depthmap_batch_input_dir)
        for path in image_list:
            try:
                inputimages.append(Image.open(path))
                inputnames.append(path)

                custom_depthmap = None
                if depthmap_batch_reuse:
                    basename = Path(path).stem
                    # Custom names are not used in samples directory
                    if outpath != backbone.get_opt('outdir_extras_samples', None):
                        # Possible filenames that the custom depthmaps may have
                        name_candidates = [f'{basename}-0000.{backbone.get_opt("samples_format", "png")}',  # current format
                                           f'{basename}.png',  # human-intuitive format
                                           f'{Path(path).name}']  # human-intuitive format (worse)
                        for fn_cand in name_candidates:
                            path_cand = os.path.join(outpath, fn_cand)
                            if os.path.isfile(path_cand):
                                custom_depthmap = Image.open(os.path.abspath(path_cand))
                                break
                inputdepthmaps.append(custom_depthmap)
            except Exception as e:
                print(f'Failed to load {path}, ignoring. Exception: {str(e)}')
        inputdepthmaps_n = len([1 for x in inputdepthmaps if x is not None])
        print(f'{len(inputimages)} images will be processed, {inputdepthmaps_n} existing depthmaps will be reused')

    gen_obj = core_generation_funnel(outpath, inputimages, inputdepthmaps, inputnames, inputs, backbone.gather_ops())

    # Saving images
    img_results = []
    results_total = 0
    inpainted_mesh_fi = mesh_simple_fi = None
    msg = ""  # Empty string is never returned
    while True:
        try:
            input_i, type, result = next(gen_obj)
            results_total += 1
        except StopIteration:
            # TODO: return more info
            msg = '<h3>Successfully generated</h3>' if results_total > 0 else \
                '<h3>Successfully generated nothing - please check the settings and try again</h3>'
            break
        except Exception as e:
            msg = format_exception(e)
            break
        if type == 'simple_mesh':
            mesh_simple_fi = result
            continue
        if type == 'inpainted_mesh':
            inpainted_mesh_fi = result
            continue
        if not isinstance(result, Image.Image):
            print(f'This is not supposed to happen! Somehow output type {type} is not supported! Input_i: {input_i}.')
            continue
        img_results += [(input_i, type, result)]

        if inputs["save_outputs"]:
            try:
                basename = 'depthmap'
                if depthmap_mode == '2' and inputnames[input_i] is not None:
                    if outpath != backbone.get_opt('outdir_extras_samples', None):
                        basename = Path(inputnames[input_i]).stem
                suffix = "" if type == "depth" else f"{type}"
                backbone.save_image(result, path=outpath, basename=basename, seed=None,
                           prompt=None, extension=backbone.get_opt('samples_format', 'png'), short_filename=True,
                           no_prompt=True, grid=False, pnginfo_section_name="extras",
                           suffix=suffix)
            except Exception as e:
                if not ('image has wrong mode' in str(e) or 'I;16' in str(e)):
                    raise e
                print('Catched exception: image has wrong mode!')
                traceback.print_exc()

    # Deciding what mesh to display (and if)
    display_mesh_fi = None
    if backbone.get_opt('depthmap_script_show_3d', True):
        display_mesh_fi = mesh_simple_fi
        if backbone.get_opt('depthmap_script_show_3d_inpaint', True):
            if inpainted_mesh_fi is not None and len(inpainted_mesh_fi) > 0:
                display_mesh_fi = inpainted_mesh_fi
    return map(lambda x: x[2], img_results), inpainted_mesh_fi, display_mesh_fi, msg.replace('\n', '<br>')



if __name__ == '__main__':
    res = run_generate(inputs)
    print(res)
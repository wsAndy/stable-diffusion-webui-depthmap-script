import rembg
import numpy as np
from PIL import Image
import os
from pathlib import Path

data = [ os.path.join("/code/data/img/", x ) for x in os.listdir("/code/data/img/") if x.lower().endswith('png')]

from rembg.bg import remove, new_session
bg_model_dir = os.path.join("/code/models/rem_bg")
os.makedirs(bg_model_dir, exist_ok=True)
os.environ["U2NET_HOME"] = str(bg_model_dir)

#create a new session by passing name of the model from one of the following
#["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"]
background_removal_session = new_session("isnet-general-use")
background_removal_session2 = new_session("u2net")

for item in data:
    # Load the input image
    input_image = Image.open(item)

    # Convert the input image to a numpy array
    input_array = np.array(input_image)

    # Apply background removal using rembg
    output_array = rembg.remove(input_array, session=background_removal_session, only_mask=True)
    output_array2 = rembg.remove(input_array, session=background_removal_session2, only_mask=True)

    output_array = (((output_array + output_array2) > 10).astype(np.uint8)*255);

    # Create a PIL Image from the output array
    output_image = Image.fromarray(output_array)

    # Save the output image
    os.makedirs("/code/data/img_nobg/", exist_ok=True)
    output_image.save( os.path.join("/code/data/img_nobg/", os.path.basename(item) ) )

del background_removal_session
del background_removal_session2

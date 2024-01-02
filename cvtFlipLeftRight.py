import os
import moviepy.editor as mp
from moviepy.audio.fx.all import volumex
import numpy as np

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        

def add_suffix(path, suff):
    # ext = os.path.basename(path).split('.')[-1]
    ext = 'mp4'
    name = os.path.basename(path).split('.')[:-1]
    return os.path.join( os.path.dirname(path), '.'.join(name) + "_" + suff + "." + ext )

input_folder = os.path.join(r"/datasets/sbs")
# allFiles = [ x for x in os.listdir(input_folder) ]

allFiles = ['20240102110247_FSBS.MOV' ]

for filename in allFiles:
    output_folder = input_folder

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, add_suffix(filename, "_leftright") )

    # 打开视频文件
    video_clip = mp.VideoFileClip(input_path)
    video_clip = volumex(video_clip, 0.2)
    sound = video_clip.audio

    frames = video_clip.iter_frames()

    arrs = []
    for item in frames:
        height_width_channel = item.shape
        outFrame = np.zeros( height_width_channel )
        halfW = int(height_width_channel[1]/2)

        outFrame[:, 0:halfW, :] = item[:, halfW:2*halfW, :]
        outFrame[:, halfW:2*halfW, :] = item[:, 0:halfW, :]
        arrs.append(outFrame)


    flipVideo = ImageSequenceClip(arrs, fps=video_clip.fps)
    flipVideo.audio = sound
    flipVideo.write_videofile(output_path, codec="libx264", audio_codec="aac", bitrate='8000k' )
        
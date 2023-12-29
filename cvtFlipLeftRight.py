import os
import moviepy.editor as mp
import numpy as np

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        

def add_suffix(path, suff):
    ext = os.path.basename(path).split('.')[-1]
    name = os.path.basename(path).split('.')[:-1]
    return os.path.join( os.path.dirname(path), '.'.join(name) + "_" + suff + "." + ext )

input_folder = os.path.join(r"/code/data")
filename = "lion.mp4"

output_folder = input_folder

input_path = os.path.join(input_folder, filename)
output_path = os.path.join(output_folder, add_suffix(filename, "_leftright") )

# 打开视频文件
video_clip = mp.VideoFileClip(input_path)
sound = video_clip.audio

frames = [x for x in list(video_clip.iter_frames())]

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
flipVideo.write_videofile(output_path, codec="libx264", audio_codec="aac", bitrate='4000k' )
    
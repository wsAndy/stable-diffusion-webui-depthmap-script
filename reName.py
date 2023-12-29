
import os
import datetime


path = os.path.join(r"/code/data/1229")

files = [ os.path.join(path, x) for x in os.listdir(path)]


count = 0
today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for i in files:
    ext = os.path.basename(i).split('.')[-1]
    # out = "{0:06d}_{1}.{2}".format(count, today, ext)
    out = "{0:06d}.{1}".format(count, ext)

    os.rename(i, os.path.join(path, out) )

    count = count + 1



# from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mp

def add_suffix(path, suff):
    ext = os.path.basename(path).split('.')[-1]
    name = os.path.basename(path).split('.')[:-1]
    return os.path.join( os.path.dirname(path), '.'.join(name) + "_" + suff + "." + ext )


def resize_videos(input_folder, output_folder, max_width=1920, max_height=1080):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有视频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # 添加其他视频格式的扩展名
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, add_suffix(filename, "resize") )

            # 打开视频文件
            video_clip = mp.VideoFileClip(input_path)

            # 检查视频尺寸是否超过指定大小
            if video_clip.size[0] * video_clip.size[1] >= max_width * max_height:
                # 调整视频尺寸并保存
                resized_clip = video_clip.resize(width=int(video_clip.size[0]/1.5), height=int(video_clip.size[1]/1.5) )
                resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            else:
                if input_folder == output_folder:
                    pass
                # 如果尺寸未超过指定大小，直接复制文件
                # shutil.copyfile(input_path, output_path)


resize_videos(path, path)

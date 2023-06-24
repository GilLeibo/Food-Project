import torch
import torchvision
import pandas as pd

# set input file
file_name = "pancake1"

# paths
result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"
input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(result_excel_path, header=None)
new_df = pd.DataFrame()

# load video and get frames
torchvision.set_video_backend("pyav")
video_path = input_file_path
video = torchvision.io.VideoReader(video_path, "video")
for index, frame in enumerate(video):
    img = frame['data'].float()
    r, g, b = torch.mean(img, dim=[1, 2])
    col = df.iloc[:, index]
    for channel_mean in [r, g, b]:
        col = col.append(pd.Series(channel_mean.numpy()))
    new_df = pd.concat([new_df, col], axis=1)

# save to Excel file
new_df.to_excel(result_excel_path, index=None, header=False)

import torch
import torchvision
import pandas as pd
from tqdm import tqdm


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


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
for index, frame in enumerate(tqdm(video)):
    img = frame['data'].float()
    img_hsv = torch.squeeze(rgb2hsv_torch(torch.unsqueeze(img, 0)))
    r, g, b = torch.mean(img, dim=[1, 2])
    h, s, v = torch.mean(img_hsv, dim=[1, 2])
    col = df.iloc[:, index]
    for channel_mean in [r, g, b]:
        col = pd.concat([col, pd.Series(channel_mean.numpy())])
    for hsv_mean in [h, s, v]:
        col = pd.concat([col, pd.Series(hsv_mean.numpy())])
    new_df = pd.concat([new_df, col], axis=1)

# save to Excel file
new_df.to_excel(result_excel_path, index=None, header=False)

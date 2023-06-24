import math
import torch
import torchvision
import pandas as pd
import subprocess
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as F

# set input file
file_name = "pancake1_grabcut"

# paths
result_csv_path = "/home/gilnetanel/Desktop/results/" + file_name + ".csv"
result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"
input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"

# delete current excel+csv results files for the given file name
cmd = 'rm ' + result_excel_path
cmd2 = 'rm ' + result_csv_path
subprocess.run(cmd, shell=True)
subprocess.run(cmd2, shell=True)

# create new results file and print empty column to it
cmd = 'touch ' + result_csv_path
cmd2 = 'printf "Empty\n" >> ' + result_csv_path
subprocess.run(cmd, shell=True)
subprocess.run(cmd2, shell=True)

# cuda memory handling:
# print(torch.cuda.memory_stats(0))
# print(torch.cuda.memory_summary(0))
torch.cuda.empty_cache()

# load dinov2 model (uncomment your desierd model):
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# move model to cuda
dinov2_vitl14.cuda()
dinov2_vitl14.eval()

# load video and get frames
torchvision.set_video_backend("pyav")
video_path = input_file_path
video = torchvision.io.VideoReader(video_path, "video")
for frame in tqdm(video):
    # show frame
    # img = torchvision.transforms.ToPILImage()(frame['data'])
    # img.show()

    # TODO: make image cropping generic according to input video
    cropped_img = F.crop(img=frame['data'], top=160, left=0, height=192, width=640)
    # img = torchvision.transforms.ToPILImage()(cropped_img)
    # img.show()

    # resize frame according to patches size and set dtype
    patch_size = 14  # as defined in assert in the model
    resize_height = (math.ceil(cropped_img.size(dim=1) / patch_size)) * patch_size
    resize_width = (math.ceil(cropped_img.size(dim=2) / patch_size)) * patch_size
    transform = torch.nn.Sequential(
        T.Resize((resize_height, resize_width), antialias=True),
        T.ConvertImageDtype(torch.float32)
    )
    transformed_frame = transform(cropped_img)
    # show frame
    # img = torchvision.transforms.ToPILImage()(transformed_frame)
    # img.show()

    # make inference and save embeddings to csv file
    ready_frame = torch.unsqueeze(transformed_frame, 0)
    ready_frame = ready_frame.cuda()
    output = dinov2_vitl14(ready_frame)  # inference
    # save embedding
    output_np = output.cpu().detach().numpy()  # convert to Numpy array
    output_df = pd.DataFrame(output_np).transpose()  # convert to dataframe
    saved_df = pd.read_csv(result_csv_path, header=0)
    merged_df = pd.concat([saved_df, output_df], axis=1)
    merged_df.to_csv(result_csv_path, index=False)

# remove first column and save to Excel file
saved_df = pd.read_csv(result_csv_path, header=0)
saved_df.drop(saved_df.columns[0], axis=1, inplace=True)
saved_df.to_csv(result_csv_path, index=False)
saved_df.to_excel(result_excel_path, index=None, header=False)

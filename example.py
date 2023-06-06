import torch
import torchvision
import numpy as np
import pandas as pd

#stats = torch.cuda.memory_stats(0)
#print(torch.cuda.memory_summary(0))
torch.cuda.empty_cache()
#load dinov2 model
#dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vitb14.cuda() #move model to cuda

torchvision.set_video_backend("pyav")
video_path = "/home/gilnetanel/Desktop/input/basketball.mp4" #data directory path
video = torchvision.io.VideoReader(video_path, "video")
frames = []
for frame in video:
    resized_frame = torchvision.transforms.functional.resize(frame['data'], [714, 1274])
    resized_frame2 = torchvision.transforms.functional.convert_image_dtype(resized_frame, torch.float32)
    frames.append(resized_frame2)
dataloader = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
for image in dataloader:
    image = image.cuda() #move data to cuda
    output = dinov2_vitb14(image) #inference
    output.cpu()
    #save embedding as csv file
    output_np = output.cpu().detach().numpy() #convert to Numpy array
    df = pd.DataFrame(output_np).transpose() #convert to a dataframe
    df_1 = pd.read_csv("/home/gilnetanel/Desktop/results.csv", header=None)
    new_pd = pd.concat([df_1, df], axis=1)
    new_pd.to_csv("/home/gilnetanel/Desktop/results.csv", index=False)
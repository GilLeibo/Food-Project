import torch
import torchvision
import numpy as np
import pandas as pd

dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') #load dinov2 model

torchvision.set_video_backend("pyav")
video_path = "/home/gilnetanel/Desktop/input/basketball.mp4" #data directory path
video = torchvision.io.VideoReader(video_path, "video")
frames = []
for frame in video:
    frames.append(frame['data'])
dataloader = torch.utils.data.DataLoader(frames, batch_size=32, shuffle=False)
images, labels = next(iter(dataloader))
images, labels = images.cuda(), labels.cuda() #move data to cuda
dinov2_vitg14.cuda() #move model to cuda

output = dinov2_vitg14(images) #inference

#save embedding as csv file
output.cpu()
output_np = output.cpu().detach().numpy() #convert to Numpy array
df = pd.DataFrame(output_np) #convert to a dataframe
df.to_csv("/home/gilnetanel/Desktop/results.csv",index=False) #save to file
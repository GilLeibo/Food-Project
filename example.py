import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') #load dinov2 model

data_dir = '/home/gilnetanel/Desktop/images' #data directory path
transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]) #data transform
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
images, labels = next(iter(dataloader))
images, labels = images.cuda(), labels.cuda() #move data to cuda
dinov2_vitg14.cuda() #move model to cuda

output = dinov2_vitg14(images) #inference

#save embedding as csv file
output.cpu()
output_np = output.cpu().detach().numpy() #convert to Numpy array
df = pd.DataFrame(output_np) #convert to a dataframe
df.to_csv("/home/gilnetanel/Desktop/results.csv",index=False) #save to file
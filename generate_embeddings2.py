import torch
import torchvision
import pandas as pd
import subprocess
from tqdm import tqdm

# delete current results file
cmd = 'rm /home/gilnetanel/Desktop/results.csv'
subprocess.run(cmd, shell=True)

# create new results file and print empty column to it
cmd = 'touch /home/gilnetanel/Desktop/results.csv'
subprocess.run(cmd, shell=True)
cmd = 'printf "Empty\n" >> ' + '/home/gilnetanel/Desktop/results.csv'
subprocess.run(cmd, shell=True)

# cuda memory handling:
#print(torch.cuda.memory_stats(0))
#print(torch.cuda.memory_summary(0))
torch.cuda.empty_cache()

# load dinov2 model (uncomment your desierd model):
#dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vitl14.cuda() #move model to cuda
dinov2_vitl14.eval()

# load video and get frames
torchvision.set_video_backend("pyav")
video_path = "/home/gilnetanel/Desktop/input/burned_panckake1.mp4"
video = torchvision.io.VideoReader(video_path, "video")
for frame in tqdm(video):
    frames = []
    resized_frame = torchvision.transforms.functional.resize(frame['data'], [350, 630])
    resized_frame_and_converted_dtype = torchvision.transforms.functional.convert_image_dtype(resized_frame, torch.float32)
    frames.append(resized_frame_and_converted_dtype)

    # iterate over images, make inference, and save embeddings to csv file
    dataloader = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
    for image in dataloader:
        image = image.cuda()
        output = dinov2_vitl14(image) #inference
        #save embedding
        output_np = output.cpu().detach().numpy() #convert to Numpy array
        output_df = pd.DataFrame(output_np).transpose() #convert to dataframe
        saved_df = pd.read_csv("/home/gilnetanel/Desktop/results.csv", header=0)
        merged_df = pd.concat([saved_df, output_df], axis=1)
        merged_df.to_csv("/home/gilnetanel/Desktop/results.csv", index=False)

# remove first column and save to Excel file
saved_df = pd.read_csv("/home/gilnetanel/Desktop/results.csv", header=0)
saved_df.drop(saved_df.columns[0], axis=1, inplace=True)
saved_df.to_csv("/home/gilnetanel/Desktop/results.csv", index=False)
saved_df.to_excel('/home/gilnetanel/Desktop/results.xlsx', index=None, header=False)
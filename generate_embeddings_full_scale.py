import math
import torch
import torchvision
import pandas as pd
import subprocess
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as F

"""
Generate embeddings with dinov2 model to input video and save to excel file.
"""
def generate_embeddings(file_name):
    print("Started to generate embeddings of file name: {} \n".format(file_name))

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
        # show frame:
        # img = torchvision.transforms.ToPILImage()(frame['data'])
        # img.show()

        # crop image if necessary:
        cropped_img = frame['data']
        # cropped_img = F.crop(img=frame['data'], top=160, left=0, height=192, width=640)
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
        # show frame:
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

    print("Finished to generate embeddings of file name: {} \n".format(file_name))


"""
Convert input tensor from rgb format to hsv format.
"""
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


"""
Add to each embedding vector values of: mean r,g,b and mean h,s,v of the frame tensor.
"""
def add_means_to_embeddings(file_name):
    print("Started to add mean values to embeddings of file name: {} \n".format(file_name))

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
        img_hsv = torch.squeeze(rgb2hsv_torch(torch.unsqueeze(img, 0)))     # convert img to hsv format
        r, g, b = torch.mean(img, dim=[1, 2])   # calc mean of r,g,b values from img
        h, s, v = torch.mean(img_hsv, dim=[1, 2])   # calc mean of h,s,v values from img_hsv

        # add mean values to embedding vector
        col = df.iloc[:, index]
        for channel_mean in [r, g, b]:
            col = pd.concat([col, pd.Series(channel_mean.numpy())])
        for hsv_mean in [h, s, v]:
            col = pd.concat([col, pd.Series(hsv_mean.numpy())])
        new_df = pd.concat([new_df, col], axis=1)

    # save to Excel file
    new_df.to_excel(result_excel_path, index=None, header=False)

    print("Finished to add mean values to embeddings of file name: {} \n".format(file_name))


def main():
    # set input file
    input_file = "pancake1"

    generate_embeddings(input_file)
    add_means_to_embeddings(input_file)


if __name__ == "__main__":
    main()

import math

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import subprocess


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size // 4),
            nn.ReLU(),
            nn.Linear(embedding_size // 4, 3),
            nn.ReLU(),
            nn.Linear(3, embedding_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) + x[:, :x.shape[1] // 2]
        return logits


def get_model(model_name):
    match model_name:
        case "dinov2_vits14":
            dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            return dinov2_vits14
        case "dinov2_vitb14":
            dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            return dinov2_vitb14
        case "dinov2_vitl14":
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            return dinov2_vitl14
        case "dinov2_vitg14":
            dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            return dinov2_vitg14


def calc_score(metric, vector1, vector2):
    match metric:
        case "L1_norm":
            vector1 = torch.unsqueeze(vector1, 0)
            vector2 = torch.unsqueeze(vector2, 0)
            return (torch.cdist(vector1, vector2, p=1)).item()
        case "L2_norm":
            vector1 = torch.unsqueeze(vector1, 0)
            vector2 = torch.unsqueeze(vector2, 0)
            return (torch.cdist(vector1, vector2, p=2)).item()
        case "cosine_similarity":
            return (cosine_similarity(vector1, vector2)).item()


trained_model_name_dict = {
    "self_videos": "embeddings_only_self_videos.zip",
    "youtube_videos": "embeddings_only_youtube_videos.zip",
    "all_videos": "embeddings_only_all_videos.zip"
}

# input size for the NeuralNetwork. Default is embeddings from dinov2_vitb14 with size of 768
embedding_model_name = "dinov2_vitb14"
embedding_size = 768
embedding_format = "embeddings_only"

if __name__ == "__main__":

    # configure settings
    input_format = "all_videos"
    embedding_model = "dinov2_vitb14"
    gap_to_calc_embedding = 90
    num_frames_to_average_threshold = 50

    assert gap_to_calc_embedding >= num_frames_to_average_threshold

    # init values
    trained_model_name = trained_model_name_dict.get(input_format)
    trained_model_path = "/home/gilnetanel/Desktop/trained_models/" + trained_model_name

    # read reference embedding
    reference_embedding_excel_path = ("/home/gilnetanel/Desktop/ROC/" + input_format + "/" + embedding_model + "_"
                                      + input_format + "_extended_reference_embedding.xlsx")
    reference_embedding = (pd.read_excel(reference_embedding_excel_path, header=None)).transpose()
    reference_embedding = (torch.tensor(reference_embedding.values)).to(torch.float32)

    for metric, threshold in zip(["L2_norm", "cosine_similarity"], [18.307222366333, 0.465407758951187]):

        # delete old directories and create new ones
        cmd1 = 'rm -r /home/gilnetanel/Desktop/predict/' + metric
        cmd2 = 'mkdir -p /home/gilnetanel/Desktop/predict/' + metric
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        for input_file in ["cheese", "pizza3", "pizza4", "sandwich", "egg1_full", "pancake1_zoomed"]:
            input_file_path = "/home/gilnetanel/Desktop/input/" + input_file + ".mp4"

            # cuda memory handling:
            torch.cuda.empty_cache()

            # load models and move to cuda
            trained_model = NeuralNetwork()
            trained_model.load_state_dict(torch.load(trained_model_path))
            trained_model.cuda()
            trained_model.eval()
            embedding_model = get_model(embedding_model_name)
            embedding_model.cuda()
            embedding_model.eval()

            # load video and get frames
            torchvision.set_video_backend("pyav")
            video_path = input_file_path
            video = torchvision.io.VideoReader(video_path, "video")

            # the dataframes to collect embeddings
            embeddings = pd.DataFrame()
            future_embeddings = pd.DataFrame()

            for frame_num, frame in enumerate(tqdm(video)):
                cropped_img = frame['data']

                # resize frame according to patches size and set dtype
                patch_size = 14  # as defined in assert in the model
                resize_height = (math.ceil(cropped_img.size(dim=1) / patch_size)) * patch_size
                resize_width = (math.ceil(cropped_img.size(dim=2) / patch_size)) * patch_size
                transform = torch.nn.Sequential(
                    T.Resize((resize_height, resize_width), antialias=True),
                    T.ConvertImageDtype(torch.float32)
                )
                transformed_frame = transform(cropped_img)
                ready_frame = torch.unsqueeze(transformed_frame, 0)

                # make inference
                embedding = embedding_model(ready_frame.cuda())  # inference

                # move to cpu
                embedding = pd.DataFrame(embedding.cpu().detach().numpy()).transpose()
                ready_frame.cpu()

                # concat embedding to embeddings dataFrame
                embeddings = pd.concat([embeddings, embedding], axis=1)
                if frame_num >= gap_to_calc_embedding:

                    # get embedding to do calc differentiation vector
                    differentiation_embedding_frame = frame_num - gap_to_calc_embedding
                    embedding2 = embeddings.iloc[:, differentiation_embedding_frame]

                    # calc differentiation embedding and concat to embedding
                    subtract = lambda s1, s2: s1.subtract(s2)
                    differences_embedding = embedding.combine(pd.DataFrame(embedding2), subtract)
                    merged_embedding = pd.concat([embedding, differences_embedding], axis=0)
                    X = torch.transpose((torch.tensor(merged_embedding.to_numpy())).to(torch.float32), 0, 1)

                    # predict
                    future_embedding = trained_model(X.cuda())

                    # concat
                    future_embedding = pd.DataFrame(future_embedding.cpu().detach().numpy()).transpose()
                    future_embeddings = pd.concat([future_embeddings, future_embedding], axis=1)

                    # calc future_embeddings score
                    future_embeddings_size = future_embeddings.shape[1]
                    if future_embeddings_size >= num_frames_to_average_threshold:

                        embeddings_for_score = future_embeddings.iloc[:,
                                               future_embeddings_size - num_frames_to_average_threshold:future_embeddings_size]
                        embeddings_for_score = embeddings_for_score.mean(axis=1)
                        embeddings_for_score = (torch.tensor(embeddings_for_score)).to(torch.float32)
                        embeddings_for_score = torch.unsqueeze(embeddings_for_score, 0)
                        score = calc_score(metric, reference_embedding, embeddings_for_score)

                        if (metric == "cosine_similarity" and score >= threshold) or (metric == "L2_norm" and score <= threshold):
                            # show frame:
                            img = torchvision.transforms.ToPILImage()(frame['data'])
                            # img.show()
                            image_save_path = ("/home/gilnetanel/Desktop/predict/" + metric + "/" + input_format + "_" +
                                               metric + "_" + input_file + "_" + str(frame_num) + ".png")
                            img.save(image_save_path)
                            print("Burned frame for file_name {} using metric {} is: {}.".format(input_file, metric, frame_num))
                            break

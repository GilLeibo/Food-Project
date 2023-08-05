import math

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


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


# input size for the NeuralNetwork. Default is embeddings from dinov2_vitb14 with size of 768
embedding_model_name = "dinov2_vitb14"
embedding_size = 768
embedding_format = "embeddings_only"

# value format: (trained_model_name, file_name of video to predict, threshold)
trained_model_metadata = {
    "self_videos": ("embeddings_only_self_videos.zip", "egg1_full", 0.015145331479015),
    "youtube_videos": ('embeddings_only_youtube_videos.zip', "pizza3", 0.00999994077971596)
}

if __name__ == "__main__":

    # configure settings
    trained_model_data = "self_videos"
    gap_to_calc_embedding = 90
    num_frames_to_average_threshold = 50

    assert gap_to_calc_embedding >= num_frames_to_average_threshold

    # init values
    trained_model_name, input_file, threshold = trained_model_metadata.get(trained_model_data)

    # paths
    trained_model_path = "/home/gilnetanel/Desktop/trained_models/" + trained_model_name
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

            # calc future_embeddings means
            future_embeddings_size = future_embeddings.shape[1]
            if future_embeddings_size >= num_frames_to_average_threshold:

                embeddings_for_threshold = future_embeddings.iloc[:,
                                           future_embeddings_size - num_frames_to_average_threshold:future_embeddings_size]
                embeddings_means = embeddings_for_threshold.mean(axis=0)
                predicted_mean = embeddings_means.mean()

                if predicted_mean >= threshold:
                    # show frame:
                    img = torchvision.transforms.ToPILImage()(frame['data'])
                    img.show()
                    image_save_path = "/home/gilnetanel/Desktop/predict/" + input_file + "_" + str(frame_num) + ".png"
                    img.save(image_save_path)
                    print("Food is ready. Burned frame is: ", frame_num)
                    break

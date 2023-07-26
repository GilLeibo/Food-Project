import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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


def get_values_according2_embedding_format(df, embedding_format):
    embeddings_features = df.iloc[:-6, :]
    rgb_features = df.iloc[-6:-3, :]
    hsv_features = df.iloc[-3:, :]

    match embedding_format:
        case "full_embeddings":
            return df
        case "embeddings_only":
            return embeddings_features
        case "embedding_rgb":
            return pd.concat([embeddings_features, rgb_features], axis=0)
        case "embedding_hsv":
            return pd.concat([embeddings_features, hsv_features], axis=0)
        case "rgb_hsv":
            return pd.concat([rgb_features, hsv_features], axis=0)
        case "rgb":
            return rgb_features
        case "hsv":
            return hsv_features


embedding_formats_dict = {
    "1": "full_embeddings",
    "2": "embeddings_only",
    "3": "embedding_rgb",
    "4": "embedding_hsv",
    "5": "rgb_hsv",
    "6": "rgb",
    "7": "hsv"
}

# input size for the NeuralNetwork. Default is embeddings from dinov2_vitb14 with size of 768
embedding_size = 768

if __name__ == "__main__":

    # configure settings
    input_file = "dinov2_vitb14_egg1"
    model_name = "embeddings_only"
    model_save_path = "/home/gilnetanel/Desktop/trained_models/" + model_name
    embedding_format_key = "2"
    prediction_time = 60  # time gap to predicate (in seconds)
    video_fps = 30  # make sure your video was filmed in 30 fps. make sure your video in normal speed (not double)

    gap_to_prediction_frame = int(prediction_time * video_fps)

    # load model and move it to cuda
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_save_path))
    model.cuda()

    # load data of input_file
    result_excel_path = "/home/gilnetanel/Desktop/results/" + input_file + ".xlsx"
    df = pd.read_excel(result_excel_path, header=None)
    embedding_format = embedding_formats_dict.get(embedding_format_key)
    data_set = get_values_according2_embedding_format(df, embedding_format)

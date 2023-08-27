import math

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


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


def add_means_to_embedding(cropped_img, embedding):
    img = cropped_img.float()
    img_hsv = torch.squeeze(rgb2hsv_torch(torch.unsqueeze(img, 0)))  # convert img to hsv format
    r, g, b = torch.mean(img, dim=[1, 2])  # calc mean of r,g,b values from img
    h, s, v = torch.mean(img_hsv, dim=[1, 2])  # calc mean of h,s,v values from img_hsv

    # add mean values to embedding vector
    for channel_mean in [r, g, b]:
        embedding = pd.concat([embedding, pd.Series(channel_mean.numpy(), dtype=np.dtype("float"))])
    for hsv_mean in [h, s, v]:
        embedding = pd.concat([embedding, pd.Series(hsv_mean.numpy(), dtype=np.dtype("float"))])

    return embedding


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


def get_embeddings_indexes(random_values, embedding_format):
    match embedding_format:
        case "embeddings_only":
            return random_values
        case "embedding_hsv":
            new_indexes = random_values.copy()
            for hsv_index in hsv_indexes:
                new_indexes.append(hsv_index)
            return new_indexes
        case "hsv":
            return hsv_indexes


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


def plot_roc_curve(roc_curve_figure_path, input_format, file_to_predict, fpr, tpr):
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve\n Input: {}, File to predict: {}'.format(input_format, file_to_predict))
    plt.savefig(roc_curve_figure_path)
    plt.close()


def plot_scores(score_curve_figure_path, input_format, file_to_predict, scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.title('Scores\n Input: {}, File to predict: {}'.format(input_format, file_to_predict))
    plt.savefig(score_curve_figure_path)
    plt.close()


hsv_indexes = [771, 772, 773]

# TODO: ["cheese", "pizza3", "pizza4", "sandwich", "egg1_full", "pancake1_zoomed"]
input_formats_dict = {
    "self_videos": ("self_videos_embeddings_only_lr0.1_epochs800.zip",
                   "self_videos_embeddings_only_extended_reference_embedding.xlsx", "embeddings_only",
                   "cosine_similarity", 0.949752032756805,
                   [("egg1_full", 150), ("pancake1_zoomed", 100)]),
    "youtube_videos": ("youtube_videos_embedding_hsv_lr0.04_epochs800.zip",
                       "youtube_videos_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                       "cosine_similarity", 0.98322993516922,
                       [("cheese", 45), ("sandwich", 22), ("pizza3", 50), ("pizza4", 58)]),
    "all_videos": ("all_videos_embedding_hsv_lr0.07_epochs800.zip",
                           "all_videos_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                           "L1_norm", 48.3728485107422,
                           [("cheese", 45), ("sandwich", 22), ("pizza3", 50), ("pizza4", 58), ("egg1_full", 150), ("pancake1_zoomed", 100)]),
    "youtube_videos_left_parts": ("youtube_videos_left_parts_embedding_hsv_lr0.1_epochs500.zip",
                               "youtube_videos_left_parts_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                               "L1_norm", 48.3728485107422,
                               [("cheese_right_part", 45), ("sandwich_right_part", 22), ("pizza3_right_part", 50), ("pizza4_right_part", 58)]),
    "pizzas": ("pizzas_embedding_hsv_lr0.07_epochs500.zip",
                                   "pizzas_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                                   "L1_norm", 48.3728485107422,
                                   [("pizza4", 58)]),
    "pizzas_left_parts": ("pizzas_left_parts_embedding_hsv_lr0.07_epochs800.zip",
                                       "pizzas_left_parts_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                                       "L1_norm", 48.3728485107422,
                                       [("pizza4_right_part", 58)]),
    "cheese_sandwich_left_part": ("cheese_sandwich_left_part_embedding_hsv_lr0.07_epochs800.zip",
                                           "cheese_sandwich_left_part_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                                           "L1_norm", 48.3728485107422,
                                           [("cheese_sandwich_right_part", 50)]),
    "pastry_left_part": ("pastry_left_part_embedding_hsv_lr0.1_epochs800.zip",
                                               "pastry_left_part_embedding_hsv_extended_reference_embedding.xlsx", "embedding_hsv",
                                               "L1_norm", 48.3728485107422,
                                               [("pastry_right_part", 50)])
}

if __name__ == "__main__":

    # configure settings
    input_formats = ["self_videos", "youtube_videos", "all_videos", "youtube_videos_left_parts", "pizzas", "pizzas_left_parts", "cheese_sandwich_left_part", "pastry_left_part"]
    embedding_model_name = "dinov2_vitb14"
    gap_to_calc_embedding = 90
    num_frames_to_average_threshold = 50

    # assertion is needed to make calculation of average threshold possible
    assert gap_to_calc_embedding >= num_frames_to_average_threshold

    # read from Excel the random values to use as indexes for the embeddings
    random_values_excel_path = '/home/gilnetanel/Desktop/random_values/random_values.xlsx'
    random_values = pd.read_excel(random_values_excel_path, header=0, names=["threshold"], index_col=None, usecols=[1])
    random_values = random_values["threshold"].tolist()

    # load models and move to cuda
    embedding_model = get_model(embedding_model_name)
    embedding_model.cuda()
    embedding_model.eval()

    # iterate all input_formats
    for input_format in input_formats:

        # remove corresponding input_format folder if exists
        cmd1 = 'rm -r /home/gilnetanel/Desktop/predict/' + input_format
        cmd2 = 'mkdir -p /home/gilnetanel/Desktop/predict/' + input_format
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        # get input_format configuration
        trained_model_name, reference_embedding_name, embedding_format, metric, threshold, files_to_predict_metadata = input_formats_dict.get(
            input_format)

        # init
        trained_model_path = "/home/gilnetanel/Desktop/predict/" + trained_model_name
        trained_model_loaded = torch.load(trained_model_path)
        reference_embedding_excel_path = "/home/gilnetanel/Desktop/predict/" + reference_embedding_name
        reference_embedding = (pd.read_excel(reference_embedding_excel_path, header=None)).transpose()
        reference_embedding = (torch.tensor(reference_embedding.values)).to(torch.float32)
        embeddings_indexes = get_embeddings_indexes(random_values, embedding_format)
        embedding_size = len(embeddings_indexes)

        # iterate over files to predict
        for file_to_predict, time_burned in files_to_predict_metadata:

            found_burned_frame = False
            scores = []
            future_embeddings_size = 0
            file_to_predict_path = "/home/gilnetanel/Desktop/input/" + file_to_predict + ".mp4"

            # cuda memory handling:
            torch.cuda.empty_cache()

            # load models and move to cuda
            trained_model = NeuralNetwork()
            trained_model.load_state_dict(trained_model_loaded)
            trained_model.cuda()
            trained_model.eval()

            # load video and get frames
            torchvision.set_video_backend("pyav")
            video_path = file_to_predict_path
            video = torchvision.io.VideoReader(video_path, "video")
            video_fps = (video.get_metadata().get('video')).get('fps')[0]
            time_burned_frame_index = int(time_burned * video_fps)

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

                # add rgb, hsv to embedding vector
                embedding = add_means_to_embedding(cropped_img, embedding)

                # fix indexes
                embedding = embedding.transpose()
                new_col = np.arange(embedding.shape[1]).tolist()
                embedding.columns = new_col
                embedding = embedding.transpose()

                # modify embedding according to configuration
                embedding = get_values_according2_embedding_format(embedding, embedding_format)
                embedding = embedding.loc[embeddings_indexes]

                # concat embedding to embeddings dataFrame
                embeddings = pd.concat([embeddings, embedding], axis=1)

                # start calculation when have enough gap
                if frame_num >= gap_to_calc_embedding:

                    # get embedding corresponding to gap_to_calc_embedding to calc differentiation embedding
                    differentiation_embedding_frame = frame_num - gap_to_calc_embedding
                    embedding2 = embeddings.iloc[:, differentiation_embedding_frame]

                    # calc differentiation embedding and concat to embedding
                    subtract = lambda s1, s2: s1.subtract(s2)
                    differences_embedding = embedding.combine(pd.DataFrame(embedding2), subtract)
                    merged_embedding = pd.concat([embedding, differences_embedding], axis=0)
                    X = torch.transpose((torch.tensor(merged_embedding.to_numpy())).to(torch.float32), 0, 1)

                    # predict future_embedding
                    future_embedding = trained_model(X.cuda())

                    # concat future_embedding to future_embeddings
                    future_embedding = pd.DataFrame(future_embedding.cpu().detach().numpy()).transpose()
                    future_embeddings = pd.concat([future_embeddings, future_embedding], axis=1)

                    # calc future_embeddings score
                    future_embeddings_size = future_embeddings.shape[1]

                    # calc average when have enough future_embeddings
                    if future_embeddings_size > num_frames_to_average_threshold:

                        index_in_future_embeddings_to_calc_score = future_embeddings_size - 1
                        embeddings_for_score = future_embeddings.iloc[:,
                                               index_in_future_embeddings_to_calc_score - num_frames_to_average_threshold:index_in_future_embeddings_to_calc_score]
                        embeddings_for_score = embeddings_for_score.mean(axis=1)
                        embeddings_for_score = (torch.tensor(embeddings_for_score)).to(torch.float32)
                        embeddings_for_score = torch.unsqueeze(embeddings_for_score, 0)
                        score = calc_score(metric, reference_embedding, embeddings_for_score)
                        scores.append(score)

                        if (((metric == "cosine_similarity" and score >= threshold)
                             or (metric == "L2_norm" and score <= threshold)
                             or (metric == "L1_norm" and score <= threshold))
                                and found_burned_frame is False):
                            img = torchvision.transforms.ToPILImage()(frame['data'])
                            # img.show()
                            image_save_path = ("/home/gilnetanel/Desktop/predict/" + input_format + "/" +
                                               file_to_predict + "_" + str(frame_num) + ".png")
                            img.save(image_save_path)
                            print(
                                "First burned frame for file_to_predict {} is: {}.".format(file_to_predict, frame_num))
                            found_burned_frame = True

            # calc true_values
            true_values = np.array([])
            true_values = np.append(true_values,
                                    np.zeros(
                                        time_burned_frame_index - gap_to_calc_embedding - num_frames_to_average_threshold))
            true_values = np.append(true_values,
                                    np.ones(embeddings.shape[1] - time_burned_frame_index))

            # calc ROC and plot graph
            fpr, tpr, thresholds = metrics.roc_curve(true_values, np.array(scores))

            # transpose ROC values if needed
            if metric == "L1_norm" or metric == "L2_norm":
                temp = fpr
                fpr = tpr
                tpr = temp

            roc_curve_figure_path = '/home/gilnetanel/Desktop/predict/' + input_format + "/" + input_format + "_" + file_to_predict + "_roc_curve.png"
            plot_roc_curve(roc_curve_figure_path, input_format, file_to_predict, fpr, tpr)
            print("Saved {} {} ROC Figure".format(input_format, file_to_predict))

            score_curve_figure_path = '/home/gilnetanel/Desktop/predict/' + input_format + "/" + input_format + "_" + file_to_predict + "_scores.png"
            plot_scores(score_curve_figure_path, input_format, file_to_predict, scores)
            print("Saved {} {} scores Figure".format(input_format, file_to_predict))

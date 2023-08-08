import numpy
import numpy as np
import pandas as pd
import torchvision
from sklearn import metrics
import matplotlib.pyplot as plt
import subprocess
import torch
from sklearn.metrics.pairwise import cosine_similarity

"""
* In this code you need to specify a video file_name and the time when the food in the video is starting to be 
considered burned and the code produce the threshold one can use with cosine_similarity to decide when other food videos
are starting to be burned (smaller than the threshold).
* You can also set the number of seconds you want the threshold will be calculated on (number of frames is inferred from
that given time).
"""


def get_reference_embedding(input_format, embedding_model, embedding_format, roc_curve_input_format_path):

    # reference embedding to return
    final_reference_embedding = pd.DataFrame()

    for file_name, time_burned in (videos_time_burned_dict.get(input_format)).items():
        # init values
        result_excel_path = "/home/gilnetanel/Desktop/results/" + embedding_model + "_" + file_name + ".xlsx"
        input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"

        # load video to get fps and get the index of first burned_frame
        torchvision.set_video_backend("pyav")
        video = torchvision.io.VideoReader(input_file_path, "video")
        video_fps = (video.get_metadata().get('video')).get('fps')[0]
        time_burned_frame_index = int(time_burned * video_fps)

        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(result_excel_path, header=None)
        embeddings = get_values_according2_embedding_format(df, embedding_format)

        # set reference embedding
        reference_embedding = embeddings.iloc[:, time_burned_frame_index - num_frames_to_average_threshold:time_burned_frame_index]
        reference_embedding = reference_embedding.mean(axis=1)
        final_reference_embedding = pd.concat([final_reference_embedding, reference_embedding], axis=1)

    # calc final_reference_embedding
    final_reference_embedding = final_reference_embedding.mean(axis=1)

    # save final_reference_embedding to Excel
    final_reference_embedding_excel_path = roc_curve_input_format_path + "/" + embedding_model + "_" + input_format + "_final_reference_embedding.xlsx"
    cmd = 'rm ' + final_reference_embedding_excel_path
    subprocess.run(cmd, shell=True)
    final_reference_embedding.to_excel(final_reference_embedding_excel_path, index=None, header=False)
    print("Saved {} final_reference_embedding values to Excel".format(input_format))

    final_reference_embedding = (torch.tensor(final_reference_embedding)).to(torch.float32)
    final_reference_embedding = torch.unsqueeze(final_reference_embedding, 0)
    return final_reference_embedding


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


def plot_roc_curve(roc_curve_figure_path, input_format):
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve. Input format: ' + input_format)
    lgd = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.savefig(roc_curve_figure_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_scores_values(metric, scores, scores_figure_path, input_format):
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Scores {}. Input format: {}'.format(metric, input_format))
    plt.savefig(scores_figure_path)
    plt.close()


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

# value is time (in seconds) to first burned frame
videos_time_burned_dict = {
    "self_videos": {"egg2_full": 160},
    "youtube_videos": {"bagle": 50, "brocolli": 50, "burek": 50, "casserole": 20, "cheese_sandwich": 50,
                       "cheesy_sticks": 55, "cherry_pie": 50,
                       "cinabbon": 40, "cinnamon": 55, "croissant": 60, "egg": 38, "nachos": 55, "pastry": 50,
                       "pizza1": 55, "pizza2": 50},
    "all_videos": {"egg2_full": 160, "bagle": 50, "brocolli": 50, "burek": 50, "casserole": 20, "cheese_sandwich": 50,
                   "cheesy_sticks": 55, "cherry_pie": 50,
                   "cinabbon": 40, "cinnamon": 55, "croissant": 60, "egg": 38, "nachos": 55, "pastry": 50, "pizza1": 55,
                   "pizza2": 50}
}

if __name__ == '__main__':
    # set input file
    embedding_model = "dinov2_vitb14"
    embedding_format_key = "2"
    num_frames_to_average_threshold = 50

    # scores
    L1_scores = []
    L2_scores = []
    cosine_scores = []
    true_values = np.array([])

    # get embedding_format
    embedding_format = embedding_formats_dict.get(embedding_format_key)

    for input_format in ["all_videos", "youtube_videos", "self_videos"]:

        # remove corresponding ROC folder if exists
        roc_curve_input_format_path = "/home/gilnetanel/Desktop/ROC/" + input_format
        cmd1 = 'rm -r ' + roc_curve_input_format_path
        cmd2 = 'mkdir -p ' + roc_curve_input_format_path
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        # get reference embedding
        reference_embedding = get_reference_embedding(input_format, embedding_model, embedding_format, roc_curve_input_format_path)

        for file_name, time_burned in (videos_time_burned_dict.get(input_format)).items():

            # init values
            result_excel_path = "/home/gilnetanel/Desktop/results/" + embedding_model + "_" + file_name + ".xlsx"
            input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"

            # load video to get fps and get the index of first burned_frame
            torchvision.set_video_backend("pyav")
            video = torchvision.io.VideoReader(input_file_path, "video")
            video_fps = (video.get_metadata().get('video')).get('fps')[0]
            time_burned_frame_index = int(time_burned * video_fps)

            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(result_excel_path, header=None)
            embeddings = get_values_according2_embedding_format(df, embedding_format)

            # calc the scores
            for frame_num in np.arange(embeddings.shape[1]):
                if frame_num >= num_frames_to_average_threshold:
                    embeddings_to_calc = embeddings.iloc[:, frame_num - num_frames_to_average_threshold:frame_num]
                    embedding_to_calc = embeddings_to_calc.mean(axis=1)
                    embedding_to_calc_tensor = (torch.tensor(embedding_to_calc)).to(torch.float32)
                    embedding_to_calc_tensor = torch.unsqueeze(embedding_to_calc_tensor, 0)
                    for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"],
                                              [L1_scores, L2_scores, cosine_scores]):
                        score = calc_score(metric, reference_embedding, embedding_to_calc_tensor)
                        scores.append(score)

            # calc true_values
            true_values = np.append(true_values, numpy.zeros(time_burned_frame_index - num_frames_to_average_threshold))
            true_values = np.append(true_values, numpy.ones(embeddings.shape[1] - time_burned_frame_index))

        for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"], [L1_scores, L2_scores, cosine_scores]):
            # calc roc_curve values
            fpr, tpr, thresholds = metrics.roc_curve(true_values, np.array(scores))

            # save values to Excel file
            roc_curve_excel_path = roc_curve_input_format_path + "/" + embedding_model + "_" + input_format + "_" + metric + "_roc_curve.xlsx"
            cmd = 'rm ' + roc_curve_excel_path
            subprocess.run(cmd, shell=True)
            dataFrames_to_save = pd.DataFrame()
            for value in [fpr, tpr, thresholds]:
                value_df = pd.DataFrame(value)
                dataFrames_to_save = pd.concat([dataFrames_to_save, value_df], axis=1)
            dataFrames_to_save.to_excel(roc_curve_excel_path, index=None, header=["fpr", "tpr", "thresholds"])
            print("Saved {} roc_curve values to Excel".format(metric))

            # plot to roc curve graph
            plt.plot(fpr, tpr, label=metric)

        # plot roc curve
        roc_curve_figure_path = roc_curve_input_format_path + "/" + embedding_model + "_" + input_format + "_roc_curve.png"
        plot_roc_curve(roc_curve_figure_path, input_format)
        print("Saved roc_curve Figure")

        # plot scores
        for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"], [L1_scores, L2_scores, cosine_scores]):
            scores_figure_path = roc_curve_input_format_path + "/" + embedding_model + "_" + input_format + "_" + metric + "_scores.png"
            plot_scores_values(metric, scores, scores_figure_path, input_format)
            print("Saved {} scores Figure".format(metric))

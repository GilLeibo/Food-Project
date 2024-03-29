import subprocess

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity


def get_reference_embedding(input_format, embedding_model, embedding_format, roc_curve_input_format_path, embeddings_indexes):
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
        reference_embedding = embeddings.loc[embeddings_indexes]
        reference_embedding = reference_embedding.iloc[:, time_burned_frame_index - num_frames_to_average_threshold:time_burned_frame_index]
        reference_embedding = reference_embedding.mean(axis=1)
        final_reference_embedding = pd.concat([final_reference_embedding, reference_embedding], axis=1)

    # calc final_reference_embedding
    final_reference_embedding = final_reference_embedding.mean(axis=1)
    final_reference_embedding = (final_reference_embedding.reset_index()).iloc[:, 1]

    # save extended_reference_embedding to Excel
    final_reference_embedding_excel_path = roc_curve_input_format_path + "/" + input_format + "_" + embedding_format + "_extended_reference_embedding.xlsx"
    final_reference_embedding.to_excel(final_reference_embedding_excel_path, index=None, header=False)
    print("Saved {} {} extended_reference_embedding to Excel".format(input_format, embedding_format))

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


def plot_roc_curve(roc_curve_figure_path, input_format, reference_embedding_format, embedding_format):
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve\n Input: {}, Embedding_format: {}'.format(input_format, embedding_format))
    lgd = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.savefig(roc_curve_figure_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_scores_values(metric, scores, scores_figure_path, input_format, reference_embedding_format, embedding_format):
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title(
        'Scores {}\n Input: {}, Embedding_format: {}'.format(metric, input_format, embedding_format))
    plt.savefig(scores_figure_path)
    plt.close()


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


hsv_indexes = [771, 772, 773]

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
                   "pizza2": 50},
    "youtube_videos_left_parts": {"bagle_left_part": 50, "brocolli_left_part": 50, "burek_left_part": 50, "casserole_left_part": 20,
                                  "cheese_sandwich_left_part": 50, "cheesy_sticks_left_part": 55, "cherry_pie_left_part": 50,
                                  "cinabbon_left_part": 40, "cinnamon_left_part": 55, "croissant_left_part": 60, "egg_left_part": 38,
                                  "nachos_left_part": 55, "pastry_left_part": 50, "pizza1_left_part": 55, "pizza2_left_part": 50},
    "pizzas": {"pizza1": 55, "pizza2": 50, "pizza3": 50},
    "pizzas_left_parts": {"pizza1_left_part": 55, "pizza2_left_part": 50, "pizza3_left_part": 50},
    "cheese_sandwich_left_part": {"cheese_sandwich_left_part": 50},
    "pastry_left_part": {"pastry_left_part": 50}

}

if __name__ == '__main__':
    # set input file
    embedding_model = "dinov2_vitb14"
    embedding_format_keys = ["2", "4"]
    reference_embedding_formats = [
        "extended", "separate"]  # separate - separate reference embedding for each file, extended - reference embedding is the same for all files and consists of mean of all files
    input_formats = ["self_videos", "youtube_videos", "all_videos", "youtube_videos_left_parts", "pizzas", "pizzas_left_parts", "cheese_sandwich_left_part", "pastry_left_part"]
    num_frames_to_average_threshold = 50

    # scores
    L1_scores = []
    L2_scores = []
    cosine_scores = []
    true_values = np.array([])

    # read from Excel the random values to use as indexes for the embeddings
    random_values_excel_path = '/home/gilnetanel/Desktop/random_values/random_values.xlsx'
    random_values = pd.read_excel(random_values_excel_path, header=0, names=["threshold"], index_col=None, usecols=[1])
    random_values = random_values["threshold"].tolist()

    # iterate all input_formats
    for input_format in input_formats:

        # remove corresponding ROC folder if exists
        cmd1 = 'rm -r /home/gilnetanel/Desktop/ROC/' + input_format
        cmd2 = 'mkdir -p /home/gilnetanel/Desktop/ROC/' + input_format
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        # iterate all embedding_format_keys
        for embedding_format_key in embedding_format_keys:

            # get embedding_format
            embedding_format = embedding_formats_dict.get(embedding_format_key)

            # get_embeddings_indexes
            embeddings_indexes = get_embeddings_indexes(random_values, embedding_format)

            # create new folder corresponding to input_format and embedding_format
            roc_curve_input_format_path = '/home/gilnetanel/Desktop/ROC/' + input_format + '/' + embedding_format
            cmd1 = 'mkdir -p ' + roc_curve_input_format_path
            subprocess.run(cmd1, shell=True)

            # iterate all reference_embedding_formats
            for reference_embedding_format in reference_embedding_formats:

                if reference_embedding_format == "extended":
                    reference_embedding = get_reference_embedding(input_format, embedding_model, embedding_format,
                                                                  roc_curve_input_format_path, embeddings_indexes)

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
                    embeddings = embeddings.loc[embeddings_indexes]

                    if reference_embedding_format == "separate":
                        reference_embedding = embeddings.iloc[:,
                                              time_burned_frame_index - num_frames_to_average_threshold:time_burned_frame_index]
                        reference_embedding = reference_embedding.mean(axis=1)
                        reference_embedding = (reference_embedding.reset_index()).iloc[:, 1]
                        reference_embedding = (torch.tensor(reference_embedding)).to(torch.float32)
                        reference_embedding = torch.unsqueeze(reference_embedding, 0)

                    # calc the scores
                    for frame_num in np.arange(embeddings.shape[1]):
                        if frame_num >= num_frames_to_average_threshold:
                            embeddings_to_calc = embeddings.iloc[:, frame_num - num_frames_to_average_threshold:frame_num]
                            embedding_to_calc = embeddings_to_calc.mean(axis=1)
                            embedding_to_calc = (embedding_to_calc.reset_index()).iloc[:, 1]
                            embedding_to_calc_tensor = (torch.tensor(embedding_to_calc)).to(torch.float32)
                            embedding_to_calc_tensor = torch.unsqueeze(embedding_to_calc_tensor, 0)
                            for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"],
                                                      [L1_scores, L2_scores, cosine_scores]):
                                score = calc_score(metric, reference_embedding, embedding_to_calc_tensor)
                                scores.append(score)

                    # calc true_values
                    true_values = np.append(true_values,
                                            numpy.zeros(time_burned_frame_index - num_frames_to_average_threshold))
                    true_values = np.append(true_values, numpy.ones(embeddings.shape[1] - time_burned_frame_index))

                for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"],
                                          [L1_scores, L2_scores, cosine_scores]):
                    # calc roc_curve values
                    fpr, tpr, thresholds = metrics.roc_curve(true_values, np.array(scores))

                    # transpose ROC values if needed
                    if metric == "L1_norm" or metric == "L2_norm":
                        temp = fpr
                        fpr = tpr
                        tpr = temp

                    # save values to Excel file
                    roc_curve_excel_path = (
                                roc_curve_input_format_path + "/" + input_format + "_" + embedding_format + "_"
                                + reference_embedding_format + "_" + metric + "_roc_curve.xlsx")
                    dataFrames_to_save = pd.DataFrame()
                    for value in [fpr, tpr, thresholds]:
                        value_df = pd.DataFrame(value)
                        dataFrames_to_save = pd.concat([dataFrames_to_save, value_df], axis=1)
                    dataFrames_to_save.to_excel(roc_curve_excel_path, index=None, header=["fpr", "tpr", "thresholds"])
                    print("Saved {} {} {} ROC to Excel".format(input_format, embedding_format, metric))

                    # plot to roc curve graph
                    plt.plot(fpr, tpr, label=metric)

                # plot roc curve
                roc_curve_figure_path = roc_curve_input_format_path + "/" + input_format + "_" + embedding_format + "_" + reference_embedding_format + "_roc_curve.png"
                plot_roc_curve(roc_curve_figure_path, input_format, reference_embedding_format, embedding_format)
                print("Saved {} {} {} ROC Figure".format(input_format, embedding_format, reference_embedding_format))

                # plot scores
                for metric, scores in zip(["L1_norm", "L2_norm", "cosine_similarity"],
                                          [L1_scores, L2_scores, cosine_scores]):
                    scores_figure_path = (
                                roc_curve_input_format_path + "/" + input_format + "_" + embedding_format + "_"
                                + reference_embedding_format + "_" + metric + "_scores.png")
                    plot_scores_values(metric, scores, scores_figure_path, input_format, reference_embedding_format,
                                       embedding_format)
                    print("Saved {} {} {} scores Figure".format(input_format, embedding_format, metric))

import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_scores(file, embedding_format):
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file + ".xlsx"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, header=None)

    desired_df = get_values_according2_embedding_format(df, embedding_format)

    scores = []

    for i in np.arange(desired_df.shape[1]-num_frames_to_average_threshold):
        embeddings_for_score = desired_df.iloc[:, i:num_frames_to_average_threshold + i]
        embeddings_means = embeddings_for_score.mean(axis=0)
        score = embeddings_means.mean()
        scores.append(score)

    return scores


def plot_scores(scores, directory_path, model_name, embedding_format, input_format):
    # plotting the points
    for key, value in scores.items():
        plt.plot(range(np.size(value)), value, label=key)

    # naming the x-axis
    plt.xlabel('Index')
    # naming the y-axis
    plt.ylabel('Score')

    # giving a title to my graph
    plt.title('Scores\n embedding_format: ' + embedding_format + ". Input format : " + input_format)
    lgd = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")

    # function to show the plot
    # plt.show()

    plt.savefig(directory_path + '/scores_' + model_name + '_' + embedding_format + "_" + input_format + '.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')

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

input_files_dict = {
    "self_videos": (["dinov2_vitb14_egg1_full", "dinov2_vitb14_egg2_full", "dinov2_vitb14_pancake1_zoomed"]),
    "youtube_videos": (
    ["dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek", "dinov2_vitb14_casserole",
     "dinov2_vitb14_cheese",
     "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
     "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant", "dinov2_vitb14_egg",
     "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2",
     "dinov2_vitb14_pizza3",
     "dinov2_vitb14_pizza4", "dinov2_vitb14_sandwich"])
}

num_frames_to_average_threshold = 50

if __name__ == '__main__':
    # configure settings
    input_formats = ["self_videos", "youtube_videos"]
    desired_embedding_formats_keys = ["2"]
    model_name = "dinov2_vitb14"

    for input_format in input_formats:
        # init
        input_files = input_files_dict.get(input_format)

        # delete model_name directory content if exist and create a new one
        cmd1 = 'rm -r /home/gilnetanel/Desktop/scores/' + input_format
        cmd2 = 'mkdir -p /home/gilnetanel/Desktop/scores/' + input_format
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        for embedding_format_key in desired_embedding_formats_keys:
            embedding_format = embedding_formats_dict.get(embedding_format_key)

            directory_path = '/home/gilnetanel/Desktop/scores/' + input_format + '/' + embedding_format

            # create new embedding_format_value directory
            cmd = 'mkdir -p ' + directory_path
            subprocess.run(cmd, shell=True)

            scores = {}
            for file in input_files:
                values = get_scores(file, embedding_format)
                scores[file] = values

            plot_scores(scores, directory_path, model_name, embedding_format, input_format)
            print("Generated scores graph for input format: ", input_format)

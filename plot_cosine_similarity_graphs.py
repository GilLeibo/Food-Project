import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

"""
Plot each cosine_similarity in different graph
"""


def plot_cosine_similarity_separately(file_name, directory_path, model_name, embedding_format):
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, header=None)

    desired_df = get_values_according2_embedding_format(df, embedding_format)

    # Convert the DataFrame to a numpy array and transpose it
    numpy_array = (desired_df.to_numpy()).transpose()

    # Calculate cosine similarity between embedding vectors
    distances = cosine_similarity(numpy_array)

    values = distances[:][0]

    # x axis values
    x = range(np.size(values))
    # corresponding y axis values
    y = values

    # plotting the points
    plt.plot(x, y)

    # naming the x-axis
    plt.xlabel('Frame')
    # naming the y-axis
    plt.ylabel('Cosine similarity')

    # giving a title to my graph
    plt.title('Cosine similarity VS Frame\n embedding_format: ' + embedding_format + ', input: ' + file_name)

    # function to show the plot
    # plt.show()

    plt.savefig(
        directory_path + '/cosine_similarity_' + model_name + '_' + embedding_format + '_' + file_name + '.png')

    plt.close()

    return values


"""
Plot all cosine_similarities of all inputs on the same graph
"""


def plot_all_cosine_similarities(cosine_similarities, directory_path, model_name, embedding_format):
    # plotting the points
    for key, value in cosine_similarities.items():
        plt.plot(range(np.size(value)), value, label=key)

    # naming the x-axis
    plt.xlabel('Frame')
    # naming the y-axis
    plt.ylabel('Cosine similarity')

    # giving a title to my graph
    plt.title('Cosine similarity VS Frame\n embedding_format: ' + embedding_format + ', all inputs')
    plt.legend()

    # function to show the plot
    # plt.show()

    plt.savefig(directory_path + '/cosine_similarity_' + model_name + '_' + embedding_format + '_all.png')

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

if __name__ == '__main__':
    # configure settings
    input_files = ["egg1", "egg1_edge", "egg1_edge_long", "egg2", "pancake1"]
    desired_embedding_formats_keys = ["3"]
    model_name = "dinov2_vitb14"

    # delete model_name directory content if exist and create a new one
    cmd1 = 'rm -r /home/gilnetanel/Desktop/Figures/' + model_name
    cmd2 = 'mkdir -p /home/gilnetanel/Desktop/Figures/' + model_name
    subprocess.run(cmd1, shell=True)
    subprocess.run(cmd2, shell=True)

    for embedding_format_key in desired_embedding_formats_keys:
        embedding_format = embedding_formats_dict.get(embedding_format_key)

        directory_path = '/home/gilnetanel/Desktop/Figures/' + model_name + '/' + embedding_format

        # create new embedding_format_value directory
        cmd = 'mkdir -p ' + directory_path
        subprocess.run(cmd, shell=True)

        cosine_similarities = {}
        for file in input_files:
            file_name = model_name + '_' + file
            values = plot_cosine_similarity_separately(file_name, directory_path, model_name, embedding_format)
            cosine_similarities[file_name] = values
            print("generated cosine similarity graph for input: ", file_name)

        plot_all_cosine_similarities(cosine_similarities, directory_path, model_name, embedding_format)
        print("generated cosine similarity graph with all inputs")

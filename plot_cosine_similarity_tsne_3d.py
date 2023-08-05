import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
from sklearn.manifold import TSNE


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
    lgd = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")

    # function to show the plot
    # plt.show()

    plt.savefig(directory_path + '/cosine_similarity_' + model_name + '_' + embedding_format + '_all.png', bbox_extra_artists=(lgd,), bbox_inches = 'tight')

    plt.close()


"""
Plot tsne3d graphs of desired inputs
"""

def plot_tsne3d_graphs(file_name, directory_path, model_name, embedding_format):
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, header=None)

    desired_df = get_values_according2_embedding_format(df, embedding_format)

    # Convert the DataFrame to a numpy array and transpose it
    numpy_array = (desired_df.to_numpy()).transpose()

    # Apply t-SNE for dimensionality reduction to 3 dimensions
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_tsne = tsne.fit_transform(numpy_array)

    # Plot the video visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2],
                 c=np.arange(len(embeddings_tsne)), cmap='viridis')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.title('Tsne3d\n embedding_format: ' + embedding_format + ', input: ' + file_name)
    plt.savefig(directory_path + '/tsne3d_' + model_name + '_' + embedding_format + '_' + file_name + '.png')
    plt.close()
    # plt.show()


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


def plot_cosine_similarity(input_files, desired_embedding_formats_keys, model_name, input_format):
    # delete model_name directory content if exist and create a new one
    cmd1 = 'rm -r /home/gilnetanel/Desktop/Figures/Cosine/' + input_format
    cmd2 = 'mkdir -p /home/gilnetanel/Desktop/Figures/Cosine/' + input_format
    subprocess.run(cmd1, shell=True)
    subprocess.run(cmd2, shell=True)

    for embedding_format_key in desired_embedding_formats_keys:
        embedding_format = embedding_formats_dict.get(embedding_format_key)

        directory_path = '/home/gilnetanel/Desktop/Figures/Cosine/' + input_format + '/' + embedding_format

        # create new embedding_format_value directory
        cmd = 'mkdir -p ' + directory_path
        subprocess.run(cmd, shell=True)

        cosine_similarities = {}
        for file in input_files:
            values = plot_cosine_similarity_separately(file, directory_path, model_name, embedding_format)
            cosine_similarities[file] = values
            print("generated cosine similarity graph for input: " + file +
                  " with embedding format: " + embedding_format)

        plot_all_cosine_similarities(cosine_similarities, directory_path, model_name, embedding_format)
        print("generated cosine similarity graph with all inputs")


def plot_tsne3d(input_files, desired_embedding_formats_keys, model_name, input_format):
    # delete model_name directory content if exist and create a new one
    cmd1 = 'rm -r /home/gilnetanel/Desktop/Figures/Tsne3d/' + input_format
    cmd2 = 'mkdir -p /home/gilnetanel/Desktop/Figures/Tsne3d/' + input_format
    subprocess.run(cmd1, shell=True)
    subprocess.run(cmd2, shell=True)

    for embedding_format_key in desired_embedding_formats_keys:
        embedding_format = embedding_formats_dict.get(embedding_format_key)

        directory_path = '/home/gilnetanel/Desktop/Figures/Tsne3d/' + input_format + '/' + embedding_format

        # create new embedding_format_value directory
        cmd = 'mkdir -p ' + directory_path
        subprocess.run(cmd, shell=True)

        for file in input_files:
            plot_tsne3d_graphs(file, directory_path, model_name, embedding_format)
            print("generated tsne3d graph for input: " + file +
                  " with embedding format: " + embedding_format)


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
    "youtube_videos": (["dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek", "dinov2_vitb14_casserole", "dinov2_vitb14_cheese",
                       "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
                       "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant", "dinov2_vitb14_egg",
                       "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2", "dinov2_vitb14_pizza3",
                        "dinov2_vitb14_pizza4", "dinov2_vitb14_sandwich"])
}

if __name__ == '__main__':
    # configure settings
    input_formats = ["self_videos", "youtube_videos"]
    desired_embedding_formats_keys = ["2", "4", "7"]
    model_name = "dinov2_vitb14"

    for input_format in input_formats:
        # init
        input_files = input_files_dict.get(input_format)

        # plot cosine similarities graphs
        plot_cosine_similarity(input_files, desired_embedding_formats_keys, model_name, input_format)

        # plot tsne3d graphs
        plot_tsne3d(input_files, desired_embedding_formats_keys, model_name, input_format)

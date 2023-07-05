import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import subprocess


def plot_tsne3d(file_name, directory_path, model_name, embedding_format):
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
    input_files = ["egg1", "egg1_edge", "egg1_edge_long", "egg1_full", "egg2", "pancake1",
                   "pancake1_zoomed", "pancake2"]
    desired_embedding_formats_keys = ["1", "2", "3", "4", "5", "6", "7"]
    model_name = "dinov2_vitb14"

    # delete model_name directory content if exist and create a new one
    cmd1 = 'rm -r /home/gilnetanel/Desktop/Figures/Tsne3d/' + model_name
    cmd2 = 'mkdir -p /home/gilnetanel/Desktop/Figures/Tsne3d/' + model_name
    subprocess.run(cmd1, shell=True)
    subprocess.run(cmd2, shell=True)

    for embedding_format_key in desired_embedding_formats_keys:
        embedding_format = embedding_formats_dict.get(embedding_format_key)

        directory_path = '/home/gilnetanel/Desktop/Figures/Tsne3d/' + model_name + '/' + embedding_format

        # create new embedding_format_value directory
        cmd = 'mkdir -p ' + directory_path
        subprocess.run(cmd, shell=True)

        for file in input_files:
            file_name = model_name + '_' + file
            plot_tsne3d(file_name, directory_path, model_name, embedding_format)
            print("generated tsne3d graph for input: " + file_name +
                  " with embedding format: " + embedding_format)

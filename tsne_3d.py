import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


def visualize_video_embedding(embeddings):
    # Apply t-SNE for dimensionality reduction to 3 dimensions
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the video visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2],
                 c=np.arange(len(embeddings_tsne)), cmap='viridis')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.title('Video Embedding Visualization')
    plt.show()


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
    file_name = "dinov2_vitb14_egg1"
    embedding_format_key = "7"

    # paths
    result_excel_path = file_name + ".xlsx"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, header=None)

    # process df according to desired embedding format
    embedding_format = embedding_formats_dict.get(embedding_format_key)
    desired_df = get_values_according2_embedding_format(df, embedding_format)

    # Convert the DataFrame to a numpy array
    numpy_array = desired_df.to_numpy()

    visualize_video_embedding(numpy_array.transpose())





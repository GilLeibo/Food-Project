import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def xlsx_to_numpy(file_path, sheet_name):
    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(file_path, header=None)

    # Convert the DataFrame to a numpy array
    numpy_array = df.to_numpy()
    return numpy_array


def visualize_video_embedding(embeddings):
    # embeddings: numpy array of shape (embedding_dim, num_frames)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the video visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=np.arange(len(embeddings_tsne)), cmap='viridis')
    plt.colorbar(label='Time (frames)')
    plt.title('Video Embedding Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

if __name__ == '__main__':
    # set input file
    file_name = "pancake1"

    # paths
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"

    numpy_array = xlsx_to_numpy(result_excel_path, 'Sheet1')
    print(numpy_array.shape)

    visualize_video_embedding(numpy_array)





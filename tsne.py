import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

if __name__ == '__main__':


    def xlsx_to_numpy(file_path, sheet_name):
        # Read the XLSX file into a pandas DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Convert the DataFrame to a numpy array
        numpy_array = df.to_numpy()

        return numpy_array


    file_path = 'results.xlsx'  # Specify the path to your XLSX file
    sheet_name = 'Sheet1'  # Specify the name of the sheet in the XLSX file

    numpy_array = xlsx_to_numpy(file_path, sheet_name)
    print(numpy_array.shape)



    def visualize_video_embedding(embeddings):
        # embeddings: numpy array of shape (num_frames, embedding_dim)

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

    visualize_video_embedding(numpy_array)




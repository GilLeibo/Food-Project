import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


if __name__ == '__main__':
    file_path = 'burned_panckake1_results.xlsx'  # Specify the path to your XLSX file

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(file_path, header=None)

    # Convert the DataFrame to a numpy array
    numpy_array = df.to_numpy()

    # Set the number of clusters (burned and unburned)
    num_clusters = 3

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(numpy_array)

    # Get the labels assigned to each frame
    labels = kmeans.labels_

    # Determine which cluster corresponds to burned food frames
    burned_cluster = np.argmax(np.bincount(labels))

    # Find the indices of burned food frames
    burned_frame_indices = np.where(labels == burned_cluster)[0]

    # Create a DataFrame to store the burned frame indices
    df = pd.DataFrame({"Burned Frame Indices": burned_frame_indices})

    # Save the DataFrame to an Excel file
    df.to_excel("burned_frames.xlsx", index=False)

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
from sklearn.cluster import KMeans

if __name__ == '__main__':


    def xlsx_to_numpy(file_path, sheet_name):
        # Read the XLSX file into a pandas DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Convert the DataFrame to a numpy array
        numpy_array = df.to_numpy()

        return numpy_array


    file_path = 'burned_panckake1_results.xlsx'  # Specify the path to your XLSX file
    sheet_name = 'Sheet1'  # Specify the name of the sheet in the XLSX file

    numpy_array = xlsx_to_numpy(file_path, sheet_name)


    # Calculate cosine distances between embedding vectors
    distances = cosine_distances(numpy_array)

    # Set a threshold value to determine burned food frames
    threshold = 0.7  # Adjust the threshold as per your needs

    # Find the indices of burned food frames
    burned_frame_indices = np.where(distances > threshold)[0]

    # Create a DataFrame to store the burned frame indices
    df = pd.DataFrame({"Burned Frame Indices": burned_frame_indices})

    # Save the DataFrame to an Excel file
    df.to_excel("burned_frames_cosine.xlsx", index=False)



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

if __name__ == '__main__':
    # set input file
    file_name = "egg1"

    file_path = file_name+'.xlsx'

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Convert the DataFrame to a numpy array
    numpy_array = df.to_numpy()

    # Calculate cosine distances between embedding vectors
    distances = cosine_distances(numpy_array)

    # Set a threshold value to determine burned food frames
    threshold = 0.7  # Adjust the threshold as per your needs

    # Find the indices of burned food frames
    burned_frame_indices = np.where(distances > threshold)[0]

    # Create a DataFrame to store the burned frame indices
    df = pd.DataFrame({"Burned Frame Indices": burned_frame_indices})

    # Save the DataFrame to an Excel file
    df.to_excel(file_name + "_cosine.xlsx", index=False)



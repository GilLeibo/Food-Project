import pandas as pd
import numpy as np
import torchvision
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    # set input file
    file_name = "pancake1"

    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"

    # set threshold
    threshold = 0.9893589940719247

    # load video and get the index of first burned_frame
    input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"
    torchvision.set_video_backend("pyav")
    video_path = input_file_path
    video = torchvision.io.VideoReader(video_path, "video")
    video_fps = (video.get_metadata().get('video')).get('fps')[0]

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, sheet_name='Sheet1')

    # Convert the DataFrame to a numpy array and transpose it
    numpy_array = (df.to_numpy()).transpose()

    # Calculate cosine similarity between embedding vectors
    distances = cosine_similarity(numpy_array)

    # set time in seconds to average threshold
    time_in_seconds_for_threshold = 3

    num_frames_to_average_threshold = int(video_fps * time_in_seconds_for_threshold)
    array = distances[:][0]

    array.shape


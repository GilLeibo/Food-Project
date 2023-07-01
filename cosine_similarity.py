import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torchvision

if __name__ == '__main__':
    # set input file
    file_name = "egg1"

    file_path = file_name+'.xlsx'

    # Set the time in the video (in seconds) from which the food considered as burned
    time_burned = 110

    # load video and get frames
    input_file_path = file_name + ".mp4"
    torchvision.set_video_backend("pyav")
    video_path = input_file_path
    video = torchvision.io.VideoReader(video_path, "video")
    video.seek(time_burned)
    frame = next(video)

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Convert the DataFrame to a numpy array and transpose it
    numpy_array = (df.to_numpy()).transpose()

    # Calculate cosine similarity between embedding vectors
    distances = cosine_similarity(numpy_array)


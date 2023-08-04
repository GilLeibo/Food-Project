import pandas as pd
import torchvision

"""
* In this code you need to specify a video file_name and the time when the food in the video is starting to be 
considered burned and the code produce the threshold one can use with cosine_similarity to decide when other food videos
are starting to be burned (smaller than the threshold).
* You can also set the number of seconds you want the threshold will be calculated on (number of frames is inferred from
that given time).
"""

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

# value is time (in seconds) to first burned frame
videos_time_burned_dict = {
    "self_videos": ("egg2_full", 160),
    "youtube_videos": ("bagle", 50)
}

if __name__ == '__main__':
    # set input file
    input_format = "youtube_videos"
    embedding_model = "dinov2_vitb14"
    embedding_format_key = "2"
    num_frames_to_average_threshold = 50

    # init values
    file_name, time_burned = videos_time_burned_dict.get(input_format)
    result_excel_path = "/home/gilnetanel/Desktop/results/" + embedding_model + "_" + file_name + ".xlsx"
    input_file_path = "/home/gilnetanel/Desktop/input/" + file_name + ".mp4"
    embedding_format = embedding_formats_dict.get(embedding_format_key)

    # load video to get fps and get the index of first burned_frame
    torchvision.set_video_backend("pyav")
    video = torchvision.io.VideoReader(input_file_path, "video")
    video_fps = (video.get_metadata().get('video')).get('fps')[0]
    time_burned_frame_index = int(time_burned * video_fps)

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, header=None)
    embeddings = get_values_according2_embedding_format(df, embedding_format)

    # calc the threshold
    threshold_calc_df = embeddings.iloc[:, time_burned_frame_index - num_frames_to_average_threshold:time_burned_frame_index]
    means = threshold_calc_df.mean(axis=0)
    threshold = means.mean()
    print("The threshold is: ", threshold)

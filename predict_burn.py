import math
import statistics
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size // 4),
            nn.ReLU(),
            nn.Linear(embedding_size // 4, 3),
            nn.ReLU(),
            nn.Linear(3, embedding_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) + x[:, :x.shape[1] // 2]
        return logits


def get_model(model_name):
    match model_name:
        case "dinov2_vits14":
            dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            return dinov2_vits14
        case "dinov2_vitb14":
            dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            return dinov2_vitb14
        case "dinov2_vitl14":
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            return dinov2_vitl14
        case "dinov2_vitg14":
            dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            return dinov2_vitg14



# input size for the NeuralNetwork. Default is embeddings from dinov2_vitb14 with size of 768
embedding_model_name = "dinov2_vitb14"
embedding_size = 768
embedding_format = "embeddings_only"


trained_model_metadata = {
    "self_videos": ("embeddings_only_self_videos.zip", "egg1_full", ),
    "youtube_videos": ('embeddings_only_youtube_videos.zip', "pizza3", )
}

if __name__ == "__main__":

    # configure settings
    trained_model_data = "self_videos"

    # init values
    trained_model_name, input_file, mean_reference = trained_model_metadata.get(trained_model_data)

    # paths
    trained_model_path = "/home/gilnetanel/Desktop/trained_models/" + trained_model_name
    input_file_path = "/home/gilnetanel/Desktop/input/" + input_file + ".mp4"

    # cuda memory handling:
    torch.cuda.empty_cache()

    # load models and move to cuda
    trained_model = NeuralNetwork()
    trained_model.load_state_dict(torch.load(trained_model_path))
    trained_model.cuda()
    trained_model.eval()

    embedding_model = get_model(embedding_model_name)
    embedding_model.cuda()
    embedding_model.eval()

    # load video and get frames
    torchvision.set_video_backend("pyav")
    video_path = input_file_path
    video = torchvision.io.VideoReader(video_path, "video")

    for frame in tqdm(video):
        cropped_img = frame['data']

        # resize frame according to patches size and set dtype
        patch_size = 14  # as defined in assert in the model
        resize_height = (math.ceil(cropped_img.size(dim=1) / patch_size)) * patch_size
        resize_width = (math.ceil(cropped_img.size(dim=2) / patch_size)) * patch_size
        transform = torch.nn.Sequential(
            T.Resize((resize_height, resize_width), antialias=True),
            T.ConvertImageDtype(torch.float32)
        )
        transformed_frame = transform(cropped_img)

        # make inference
        ready_frame = torch.unsqueeze(transformed_frame, 0)
        embedding = embedding_model(ready_frame.cuda())  # inference
        # predict
        future_embedding = trained_model(embedding.cuda())
        # move to cpu
        ready_frame.cpu()
        embedding.cpu()
        future_embedding.cpu()
        mean_future_embedding = statistics.mean(future_embedding)

        if abs(mean_future_embedding-mean_reference) < embedding_distances:
            # show frame:
            img = torchvision.transforms.ToPILImage()(transformed_frame)
            img.show()
            print("Food is ready")
            break

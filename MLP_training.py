import pandas as pd
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics.pairwise import cosine_distances


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), int(input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(input_size / 4), input_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        transposed_x = torch.transpose(x, 0, 1)
        logits = self.linear_relu_stack(transposed_x)
        return logits


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

# 768 is the default, embeddings_only size
input_size = 768

if __name__ == "__main__":

    # configure settings
    embedding_format_key = "2"
    file_name = "dinov2_vitb14_egg1"
    prediction_time = 60    # time gap to predicate (in seconds)
    video_fps = 30      # make sure your video was filmed in 30 fps. make sure your video in normal speed (not double)
    test_set_size = 100     # number of frames for the test set
    n_epochs = 5  # number of epochs to run
    batch_size = 100  # size of each batch

    # create dataset for training
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"
    df = pd.read_excel(result_excel_path, header=None)
    embedding_format = embedding_formats_dict.get(embedding_format_key)
    data_set = get_values_according2_embedding_format(df, embedding_format)
    input_size = data_set.shape[0]

    # input size needs to be even
    assert input_size % 2 == 0

    # creates model instance and move it to cuda
    model = NeuralNetwork()
    model.cuda()

    # load the dataset
    gap_to_prediction_frame = int(prediction_time * video_fps)
    X = data_set.iloc[:, :-gap_to_prediction_frame]
    Y = data_set.iloc[:, gap_to_prediction_frame:]

    # split the dataset into training and test sets
    Xtrain = X.iloc[:, :-test_set_size]
    Ytrain = Y.iloc[:, :-test_set_size]
    Xtest = X.iloc[:, -test_set_size:]
    Ytest = Y.iloc[:, -test_set_size:]

    # loss
    loss_func = nn.CosineEmbeddingLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # collect statistics
    train_loss = []
    test_acc = []

    batches_per_epoch = Xtrain.shape[1] // batch_size
    for epoch in range(n_epochs):
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # make sure gradient tracking is on
                model.train(True)
                # take a batch
                start = i * batch_size
                Xbatch = (torch.tensor(Xtrain.iloc[:, start:start + batch_size].to_numpy())).to(torch.float32)
                Ybatch = (torch.tensor(Ytrain.iloc[:, start:start + batch_size].to_numpy())).to(torch.float32)
                # zero gradients
                optimizer.zero_grad()
                # forward pass
                Y_pred = (torch.transpose(model(Xbatch.cuda()), 0, 1)).cpu()
                # compute the loss and its gradients
                loss = loss_func(torch.transpose(Y_pred, 0, 1), torch.transpose(Ybatch, 0, 1), torch.ones(Y_pred.shape[1]))
                train_loss.append(loss.item())
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(
                    loss=float(loss)
                )
        # evaluate model at end of epoch
        Y_pred = model(Xtest)
        acc = (Y_pred.round() == Ytest).float().mean()
        test_acc.append(float(acc))
        print(f"End of {epoch}, accuracy {acc}")

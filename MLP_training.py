import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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


class EmbeddingsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(dim=0)

    def __getitem__(self, idx):
        indices = torch.tensor([idx])
        return torch.index_select(self.X, 0, indices), torch.index_select(self.Y, 0, indices)


def get_train_test_sets(input_files, gap_to_prediction_frame, gap_to_calc_embedding, test_set_size, embedding_format):
    # init dataFrames for datasets
    X = pd.DataFrame()
    Y = pd.DataFrame()

    for input_file in input_files:
        # load data of input_file
        result_excel_path = "/home/gilnetanel/Desktop/results/" + input_file + ".xlsx"
        df = pd.read_excel(result_excel_path, header=None)
        data_set = get_values_according2_embedding_format(df, embedding_format)

        # check embedding_size is correct
        assert embedding_size == data_set.shape[0]

        # generate the dataset for input file
        X_embedding1 = data_set.iloc[:, gap_to_calc_embedding:-gap_to_prediction_frame]
        X_embedding2 = data_set.iloc[:, :-(gap_to_prediction_frame + gap_to_calc_embedding)]

        new_col = np.arange(X_embedding1.shape[1]).tolist()
        X_embedding1.columns = new_col

        subtract = lambda s1, s2: s1.subtract(s2)
        X_embedding_differences = X_embedding1.combine(X_embedding2, subtract)
        X_input_file = pd.concat([X_embedding1, X_embedding_differences])
        Y_input_file = data_set.iloc[:, gap_to_calc_embedding + gap_to_prediction_frame:]

        # add dataset of input file to the overall dataset
        X = pd.concat([X, X_input_file], axis=1)
        Y = pd.concat([Y, Y_input_file], axis=1)

    # convert dataFrames to tensors and transpose to common shapes: (num_samples, features_size)
    X = torch.transpose((torch.tensor(X.to_numpy())).to(torch.float32), 0, 1)
    Y = torch.transpose((torch.tensor(Y.to_numpy())).to(torch.float32), 0, 1)

    # split the dataset into training and test sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_set_size, random_state=42, shuffle=True)

    return Xtrain, Xtest, Ytrain, Ytest


def plot_losses(test_losses, input_format, embedding_format):
    # x axis values
    x = range(len(test_losses))

    # plotting the points
    plt.plot(x, test_losses)

    # naming the x-axis
    plt.xlabel('epoch')
    # naming the y-axis
    plt.ylabel('test loss')

    # giving a title to my graph
    plt.title('Test loss VS Epoch. Input_format: ' + input_format)

    # function to show the plot
    # plt.show()

    plt.savefig(
        "/home/gilnetanel/Desktop/trained_models/test_losses_graph_" + embedding_format + "_" + input_format + ".png")
    print("Saved test_losses_graph")

    plt.close()


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

# each value consists of: (learning_rate, number_epochs, input_files)
input_files_dict = {
    "self_videos": (0.12, 500, ["dinov2_vitb14_egg2_full"]),
    "youtube_videos": (0.05, 600, ["dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek", "dinov2_vitb14_casserole",
                       "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
                       "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant", "dinov2_vitb14_egg",
                       "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2"]),
    "all_videos": (0.04, 700, ["dinov2_vitb14_egg2_full", "dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek", "dinov2_vitb14_casserole",
                           "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
                           "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant", "dinov2_vitb14_egg",
                           "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2"])
}

# input size for the NeuralNetwork. Default is embeddings from dinov2_vitb14 with size of 768
embedding_size = 768

if __name__ == "__main__":

    # configure settings
    embedding_format_key = "2"
    input_format = "all_videos"  # self_videos - dataset is videos we filmed, youtube_videos - dataset is videos from youtube, all_videos - combination of the two
    test_set_size = 0.25  # portion of test_set size from dataset
    batch_size = 100  # size of each batch
    gap_to_prediction_frame = 450   # gap (in frames) to prediction frame
    gap_to_calc_embedding = 90  # gap (in frames) to the embedding which will be used to calc differentiation from current embedding

    # init values
    lr, n_epochs, input_files = input_files_dict.get(input_format)
    embedding_format = embedding_formats_dict.get(embedding_format_key)
    Xtrain, Xtest, Ytrain, Ytest = get_train_test_sets(input_files, gap_to_prediction_frame,
                                                       gap_to_calc_embedding, test_set_size, embedding_format)

    # set datasets and dataloaders
    train_dataset = EmbeddingsDataset(Xtrain, Ytrain)
    test_dataset = EmbeddingsDataset(Xtest, Ytest)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # creates model instance and move it to cuda
    model = NeuralNetwork()
    model.cuda()

    # loss
    loss_func = nn.CosineEmbeddingLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # statistics
    best_test_loss = float('inf')
    test_losses = []

    flatten = nn.Flatten()

    for epoch in range(n_epochs):
        # make sure gradient tracking is on
        model.train(True)
        for data in train_dataloader:
            # take a batch
            Xbatch, Ybatch = data
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            Y_pred = model(Xbatch.cuda()).cpu()
            # compute the loss and its gradients
            loss = loss_func(Y_pred, flatten(Ybatch), torch.ones(Y_pred.shape[0]))
            loss.backward()
            # update weights
            optimizer.step()
            # print progress

        # at end of epoch, evaluate model on test_set
        model.eval()
        Y_pred = model(Xtest.cuda()).cpu()
        test_loss = loss_func(Y_pred, Ytest, torch.ones(Y_pred.shape[0]))
        test_losses.append(test_loss.item())
        print(f"End of epoch {epoch}, test_loss {test_loss}")

        # track the best performance and save the model's state
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_save_path = "/home/gilnetanel/Desktop/trained_models/" + embedding_format + "_" + input_format + ".zip"
            torch.save(model.state_dict(), model_save_path)

    plot_losses(test_losses, input_format, embedding_format)

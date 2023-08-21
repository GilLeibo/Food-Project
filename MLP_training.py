import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import subprocess


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


def plot_learning_rates(input_format, embedding_format, n_epochs):
    # naming the x-axis
    plt.xlabel('Epoch')
    # naming the y-axis
    plt.ylabel('Test loss')

    # giving a title to my graph
    plt.title(
        'Test losses VS Epochs with different learning rates\n Input: ' + input_format + ", Embedding_format: " + embedding_format + ", Epochs: " + str(n_epochs))
    lgd = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    # function to show the plot
    # plt.show()

    plt.savefig(
        "/home/gilnetanel/Desktop/trained_models/" + input_format + '/' + embedding_format +
        '/' + input_format + "_" + embedding_format + "_epochs" + str(n_epochs), bbox_extra_artists=(lgd,), bbox_inches = 'tight')
    print("Saved test_losses_graph: " + input_format + " " + embedding_format + " epochs " + str(n_epochs))

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
    "1": ("full_embeddings", 774),
    "2": ("embeddings_only", 768),
    "3": ("embedding_rgb", 771),
    "4": ("embedding_hsv", 771),
    "5": ("rgb_hsv", 6),
    "6": ("rgb", 3),
    "7": ("hsv", 3)
}

input_files_dict = {
    "self_videos": ["dinov2_vitb14_egg2_full"],
    "youtube_videos": ["dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek",
                       "dinov2_vitb14_casserole",
                       "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
                       "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant",
                       "dinov2_vitb14_egg",
                       "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2"],
    "all_videos": ["dinov2_vitb14_egg2_full", "dinov2_vitb14_bagle", "dinov2_vitb14_brocolli", "dinov2_vitb14_burek",
                   "dinov2_vitb14_casserole",
                   "dinov2_vitb14_cheese_sandwich", "dinov2_vitb14_cheesy_sticks", "dinov2_vitb14_cherry_pie",
                   "dinov2_vitb14_cinabbon", "dinov2_vitb14_cinnamon", "dinov2_vitb14_croissant", "dinov2_vitb14_egg",
                   "dinov2_vitb14_nachos", "dinov2_vitb14_pastry", "dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2"],
    "youtube_videos_left_parts": ["dinov2_vitb14_bagle_left_part", "dinov2_vitb14_brocolli_left_part", "dinov2_vitb14_burek_left_part",
                       "dinov2_vitb14_casserole_left_part", "dinov2_vitb14_cheese_sandwich_left_part", "dinov2_vitb14_cheesy_sticks_left_part",
                       "dinov2_vitb14_cherry_pie_left_part", "dinov2_vitb14_cinabbon_left_part", "dinov2_vitb14_cinnamon_left_part",
                       "dinov2_vitb14_croissant_left_part", "dinov2_vitb14_egg_left_part",
                       "dinov2_vitb14_nachos_left_part", "dinov2_vitb14_pastry_left_part", "dinov2_vitb14_pizza1_left_part", "dinov2_vitb14_pizza2_left_part"],
    "pizzas": ["dinov2_vitb14_pizza1", "dinov2_vitb14_pizza2", "dinov2_vitb14_pizza3"],
    "pizzas_left_parts": ["dinov2_vitb14_pizza1_left_part", "dinov2_vitb14_pizza2_left_part", "dinov2_vitb14_pizza3_left_part"]
}


if __name__ == "__main__":

    # configure settings
    embedding_format_keys = ["2", "4"]
    input_formats = ["example"]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    n_epochs_list = [500, 600, 700]
    test_set_size = 0.25  # portion of test_set size from dataset
    batch_size = 100  # size of each batch
    gap_to_prediction_frame = 450  # gap (in frames) to prediction frame
    gap_to_calc_embedding = 90  # gap (in frames) to the embedding which will be used to calc differentiation from current embedding

    # iterate all input_formats
    for input_format in input_formats:

        # delete model_name directory content if exist and create a new one
        cmd1 = 'rm -r /home/gilnetanel/Desktop/trained_models/' + input_format
        cmd2 = 'mkdir -p /home/gilnetanel/Desktop/trained_models/' + input_format
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        input_files = input_files_dict.get(input_format)

        # iterate all embedding_format_keys
        for embedding_format_key in embedding_format_keys:
            embedding_format, embedding_size = embedding_formats_dict.get(embedding_format_key)

            cmd1 = 'mkdir -p /home/gilnetanel/Desktop/trained_models/' + input_format + '/' + embedding_format
            subprocess.run(cmd1, shell=True)

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

            # iterate n_epochs
            for n_epochs in n_epochs_list:

                # iterate learning_rates
                for lr in learning_rates:

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
                            model_save_path = (
                                    "/home/gilnetanel/Desktop/trained_models/" + input_format + '/' + embedding_format +
                                    '/' + input_format + "_" + embedding_format + "_lr" + str(lr) + "_epochs" + str(
                                n_epochs) + ".zip")
                            torch.save(model.state_dict(), model_save_path)

                    # plotting the results at end of the epochs
                    plt.plot(range(len(test_losses)), test_losses, label=lr)

                plot_learning_rates(input_format, embedding_format, n_epochs)

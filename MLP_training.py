import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


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
        logits = self.linear_relu_stack(transposed_x) + x[:, :x.shape[1] // 2]
        return logits


def plot_losses(test_losses):

    # x axis values
    x = range(len(test_losses))

    # plotting the points
    plt.plot(x, test_losses)

    # naming the x-axis
    plt.xlabel('epoch')
    # naming the y-axis
    plt.ylabel('test loss')

    # giving a title to my graph
    plt.title('Test loss VS Epoch')

    # function to show the plot
    # plt.show()

    plt.savefig("/home/gilnetanel/Desktop/trained_models/test_losses_graph.png")

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

# 768 is the default, embeddings_only size
input_size = 768

if __name__ == "__main__":

    # configure settings
    embedding_format_key = "2"
    file_name = "dinov2_vitb14_egg1"
    prediction_time = 60    # time gap to predicate (in seconds)
    video_fps = 30      # make sure your video was filmed in 30 fps. make sure your video in normal speed (not double)
    test_set_size = 100     # number of frames for the test set
    n_epochs = 200  # number of epochs to run
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

    # convert dataFrames to tensors
    Xtest = (torch.tensor(Xtest.to_numpy())).to(torch.float32)
    Ytest = (torch.tensor(Ytest.to_numpy())).to(torch.float32)

    # loss
    loss_func = nn.CosineEmbeddingLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # statistics
    best_test_loss = float('inf')
    test_losses = []

    batches_per_epoch = Xtrain.shape[1] // batch_size
    for epoch in range(n_epochs):
        # make sure gradient tracking is on
        model.train(True)
        for i in range(batches_per_epoch):
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
            # print(f"batch {i}, loss {loss.item()}")
            loss.backward()
            # update weights
            optimizer.step()
            # print progress

        # evaluate model at end of epoch on test_set
        model.eval()
        Y_pred = (torch.transpose(model(Xtest.cuda()), 0, 1)).cpu()
        test_loss = loss_func(torch.transpose(Y_pred, 0, 1), torch.transpose(Ytest, 0, 1), torch.ones(Y_pred.shape[1]))
        test_losses.append(test_loss.item())
        print(f"End of epoch {epoch}, test_loss {test_loss}")

        # track the best performance and save the model's state
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_save_path = "/home/gilnetanel/Desktop/trained_models/" + embedding_format
            torch.save(model.state_dict(), model_save_path)

    plot_losses(test_losses)

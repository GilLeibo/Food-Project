import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

"""
Plot each cosine_similarity in different graph
"""
def plot_cosine_similarity_separately(file_name):
    result_excel_path = "/home/gilnetanel/Desktop/results/" + file_name + ".xlsx"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(result_excel_path, sheet_name='Sheet1')

    # Convert the DataFrame to a numpy array and transpose it
    numpy_array = (df.to_numpy()).transpose()

    # Calculate cosine similarity between embedding vectors
    distances = cosine_similarity(numpy_array)

    values = distances[:][0]

    # x axis values
    x = range(np.size(values))
    # corresponding y axis values
    y = values

    # plotting the points
    plt.plot(x, y)

    # naming the x-axis
    plt.xlabel('Frame')
    # naming the y-axis
    plt.ylabel('Cosine similarity')

    # giving a title to my graph
    plt.title('Cosine similarity from first frame. Input: ' + file_name)

    # function to show the plot
    # plt.show()

    plt.savefig('/home/gilnetanel/Desktop/Figures/cosine_similarity_'+file_name+'.png')

    plt.close()

    return values


"""
Plot all cosine_similarities of all inputs on the same graph
"""
def plot_all_cosine_similarities(cosine_similarities):
    # plotting the points
    for key, value in cosine_similarities.items():
        plt.plot(range(np.size(value)), value, label=key)

    # naming the x-axis
    plt.xlabel('Frame')
    # naming the y-axis
    plt.ylabel('Cosine similarity')

    # giving a title to my graph
    plt.title('Cosine similarity from first frame. All inputs')
    plt.legend()

    # function to show the plot
    # plt.show()

    plt.savefig('/home/gilnetanel/Desktop/Figures/cosine_similarity_all.png')

    plt.close()


if __name__ == '__main__':
    # set input files
    input_files = ["egg1", "egg1_edge", "egg1_edge_long", "egg2", "pancake1"]
    cosine_similarities = {}

    for file in input_files:
        values = plot_cosine_similarity_separately(file)
        cosine_similarities[file] = values
        print("generated cosine similarity graph for input: ", file)

    plot_all_cosine_similarities(cosine_similarities)


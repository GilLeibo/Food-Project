import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import pandas as pd
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the Excel sheet into a Pandas DataFrame
    df = pd.read_excel('example.xlsx', header=None)

    # Extract the last 6 features
    six_features = df.iloc[-7:, :]
    first_three_features = df.iloc[-7:-3, :]
    last_three_features = df.iloc[-3:, :]

    # Perform t-SNE with 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(six_features)

    # Plot the video visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.arange(len(tsne_results)), cmap='viridis')
    plt.colorbar(label='Time (frames)')
    plt.title('Video Embedding Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()



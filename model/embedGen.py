import numpy as np
import os

# Define the data directory and the shape of embeddings
data_dir = "C:/Users/rub/Desktop/Stanford/CS230/Project/Zero-Shot-Sketch-Based-Image-Retrieval-master/Zero-Shot-Sketch-Based-Image-Retrieval-master/Dataset/"
num_labels = 50  # Adjust this to the number of labels you have
embedding_dim = 300  # Adjust this to the desired embedding dimension

# Initialize the embeddings randomly
train_label_embeddings = np.random.rand(num_labels, embedding_dim)

# Optionally, save the embeddings for future use
np.save(os.path.join(data_dir, 'train_embeddings.npy'), train_label_embeddings)
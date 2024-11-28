import numpy as np
import os

np.random.seed(123)

# Define the data directory and the shape of embeddings
data_dir = "Dataset/"
num_labels = 125  # Adjust this to the number of labels you have
embedding_dim = 300  # Adjust this to the desired embedding dimension

# Initialize the embeddings randomly
train_label_embeddings = np.random.rand(num_labels, embedding_dim)

# Optionally, save the embeddings for future use
np.save(os.path.join(data_dir, 'train_embeddings_125.npy'), train_label_embeddings)
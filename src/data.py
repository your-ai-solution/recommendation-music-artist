import os
import random
import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Set fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
RAW_DIR = "data/raw"
PREPROCESSED_DIR = "data/preprocessed"
ZIP_FILE = os.path.join(RAW_DIR, "hetrec2011-lastfm-2k.zip")
USER_ARTISTS_FILE = os.path.join(PREPROCESSED_DIR, "user_artists.dat")
USER_FRIENDS_FILE = os.path.join(PREPROCESSED_DIR, "user_friends.dat")

# Ensure preprocessed directory exists
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def unzip_data():
    """Unzips the dataset."""
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(PREPROCESSED_DIR)
    print("Dataset unzipped.")

def preprocess_interaction_matrix(R):
    """Normalizes the interaction matrix."""
    print("Normalizing interaction matrix...")
    max_value = R.data.max()  # Find the maximum value in the matrix
    if max_value > 0:
        R.data = R.data / max_value  # Normalize the data in the sparse matrix
    print(f"Interaction matrix normalized. Max value: {R.data.max()}")
    return R

def split_interaction_matrix(R, test_size=0.2):
    """Splits the interaction matrix into training and testing datasets."""
    print("Splitting interaction matrix into train and test sets...")
    rows, cols = R.nonzero()
    data = R.data
    train_idx, test_idx = train_test_split(range(len(data)), test_size=test_size, random_state=42)

    # Create training and testing matrices
    R_train = csr_matrix((data[train_idx], (rows[train_idx], cols[train_idx])), shape=R.shape)
    R_test = csr_matrix((data[test_idx], (rows[test_idx], cols[test_idx])), shape=R.shape)

    print(f"Train matrix: {R_train.shape}, Test matrix: {R_test.shape}")
    return R_train, R_test

def create_interaction_matrix():
    """Creates a user-artist interaction matrix and user-friend adjacency matrix."""
    print("Creating interaction and adjacency matrices...")

    # Load user-artists data
    user_artists = pd.read_csv(USER_ARTISTS_FILE, sep="\t")
    num_users = user_artists["userID"].max() + 1
    num_artists = user_artists["artistID"].max() + 1

    # User-artist interaction matrix
    interactions = np.array(user_artists[["userID", "artistID", "weight"]])

    # Binarize interaction weights for classification
    interactions[:, 2] = (interactions[:, 2] > 0).astype(np.float32)

    # Split the interaction data
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)

    # Save data for NCF
    np.save(os.path.join(PREPROCESSED_DIR, "train_data.npy"), train_data)
    np.save(os.path.join(PREPROCESSED_DIR, "test_data.npy"), test_data)

    print("Interaction data saved for NCF.")

def main():
    unzip_data()
    create_interaction_matrix()

if __name__ == "__main__":
    main()
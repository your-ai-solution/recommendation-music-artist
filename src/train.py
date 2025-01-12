import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

# Set fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Paths
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
TRAIN_DATA_FILE = "data/preprocessed/train_data.npy"

# Ensure results and configs directories exist
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Neural collaborative filtering model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=-1)
        return self.fc_layers(concat_embeds).squeeze()

# Dataset with negative sampling
class InteractionDataset(Dataset):
    def __init__(self, interactions, num_items, negative_samples=4):
        # Create a set of (user, item) pairs for quick lookups
        self.positive_interactions = set((int(user), int(item)) for user, item, _ in interactions)
        self.samples = []
        self.labels = []

        # Add positive samples
        for user, item, label in interactions:
            self.samples.append((int(user), int(item)))
            self.labels.append(float(label))

            # Add negative samples for each positive sample
            for _ in range(negative_samples):
                neg_item = np.random.randint(0, num_items)
                while (user, neg_item) in self.positive_interactions:  # Ensure it's truly negative
                    neg_item = np.random.randint(0, num_items)
                self.samples.append((int(user), int(neg_item)))
                self.labels.append(0.0)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user, item = self.samples[idx]
        label = self.labels[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def train_ncf(train_data, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16],
              batch_size=256, epochs=10, learning_rate=0.001, negative_samples=4):
    dataset = InteractionDataset(train_data, num_items, negative_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users, num_items, embedding_dim, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids, item_ids, labels in dataloader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model

def main():
    # Load training data
    train_data = np.load(TRAIN_DATA_FILE)

    # Define model parameters
    num_users = int(train_data[:, 0].max() + 1)
    num_items = int(train_data[:, 1].max() + 1)

    # Train the NCF model
    model = train_ncf(train_data, num_users, num_items)

    # Save the model
    torch.save(model.state_dict(), os.path.join(CONFIGS_DIR, "ncf_model.pth"))
    print("Model saved to configs/ directory.")

    # Save num_users and num_items
    with open(os.path.join(CONFIGS_DIR, "model_config.txt"), "w") as f:
        f.write(f"{num_users}\n{num_items}\n")
    print("Model configuration saved.")

if __name__ == "__main__":
    main()
import os
import random
import numpy as np
import torch
from torch import nn

# Paths
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
MODEL_FILE = os.path.join(CONFIGS_DIR, "ncf_model.pth")
FRIEND_FILE = "data/preprocessed/user_friends.dat"

# Load num_users and num_items
with open(os.path.join(CONFIGS_DIR, "model_config.txt"), "r") as f:
    num_users, num_items = map(int, f.readlines())

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

def load_model():
    """Load the trained NCF model."""
    model = NCF(num_users, num_items)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model

def load_friend_data(file_path):
    """Load user-friend relationships into a dictionary."""
    friends = {}
    with open(file_path, "r") as f:
        next(f)  # Skip the header line
        for line in f:
            user, friend = map(int, line.strip().split())
            if user not in friends:
                friends[user] = []
            friends[user].append(friend)
    return friends

def recommend(model, user, num_items, k=10, device="cpu"):
    """Generate top-k recommendations for a given user."""
    user_tensor = torch.tensor([user] * num_items, dtype=torch.long).to(device)
    item_tensor = torch.arange(num_items, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    recommended_items = torch.topk(scores, k).indices.cpu().numpy()
    return recommended_items

def main():
    print("Loading the trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)

    print("Loading friend data...")
    friends_data = load_friend_data(FRIEND_FILE)

    print("Randomly selecting users...")
    selected_users = random.sample(range(num_users), 10)  # Randomly select 10 users

    print("Generating recommendations...")
    results = []
    for user in selected_users:
        recommendations = recommend(model, user, num_items, k=10, device=device)
        friends = friends_data.get(user, [])  # Retrieve friends or empty list if none
        results.append({"user": user, "recommendations": recommendations, "friends": friends})

    # Save results to a text file
    output_file = os.path.join(RESULTS_DIR, "recommendations.txt")
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"User {result['user']}:\n")
            f.write(f"  Top-{len(result['recommendations'])} Recommendations (Artists): {list(result['recommendations'])}\n")
            f.write(f"  Friends (User IDs): {list(result['friends'])}\n\n")

    print(f"Recommendations saved to {output_file}")

if __name__ == "__main__":
    main()
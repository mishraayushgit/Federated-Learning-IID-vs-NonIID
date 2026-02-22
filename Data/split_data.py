import torch
from torchvision import datasets, transforms
from collections import defaultdict
import random

# Configuration
NUM_CLIENTS = 3
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST
train_dataset = datasets.MNIST(
    root="./Data",
    train=True,
    download=False,
    transform=transform
)

def iid_split(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    client_data = {}
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        client_data[i] = indices[start:end]

    return client_data

def non_iid_split(dataset, num_clients, classes_per_client=2):
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_data = {i: [] for i in range(num_clients)}
    classes = list(class_indices.keys())

    for client in range(num_clients):
        selected_classes = random.sample(classes, classes_per_client)
        for cls in selected_classes:
            client_data[client].extend(
                class_indices[cls][:len(class_indices[cls]) // num_clients]
            )

    return client_data

# Generate splits
iid_clients = iid_split(train_dataset, NUM_CLIENTS)
non_iid_clients = non_iid_split(train_dataset, NUM_CLIENTS)

print("IID Split:")
for k, v in iid_clients.items():
    print(f"Client {k}: {len(v)} samples")

print("\nNon-IID Split:")
for k, v in non_iid_clients.items():
    print(f"Client {k}: {len(v)} samples")

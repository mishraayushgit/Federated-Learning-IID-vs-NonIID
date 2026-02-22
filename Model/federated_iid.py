import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import OrderedDict

from cnn_model import CNNModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
NUM_CLIENTS = 3
BATCH_SIZE = 64
LOCAL_EPOCHS = 1
ROUNDS = 3

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="../Data",
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="../Data",
    train=False,
    download=False,
    transform=transform,
)

# IID Split
data_per_client = len(train_dataset) // NUM_CLIENTS
indices = list(range(len(train_dataset)))

client_indices = [
    indices[i * data_per_client:(i + 1) * data_per_client]
    for i in range(NUM_CLIENTS)
]

def get_dataloader(idxs):
    subset = Subset(train_dataset, idxs)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader):
        self.model = CNNModel().to(device)
        self.trainloader = trainloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.trainloader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"\n🔵 Round Accuracy: {accuracy*100:.2f}%\n")

        return 0.0, len(test_loader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    trainloader = get_dataloader(client_indices[int(cid)])
    return FlowerClient(trainloader)


# Strategy with accuracy aggregation
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "accuracy": sum([num_examples * m["accuracy"] for num_examples, m in metrics]) 
                    / sum([num_examples for num_examples, _ in metrics])
    }
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
    strategy=strategy,
)
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./Data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./Data",
    train=False,
    download=True,
    transform=transform
)

print("MNIST downloaded successfully!")
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

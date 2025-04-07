from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from config.config import get_config

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


train_loader = DataLoader(
    train_dataset, batch_size=get_config()["batch_size"], shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=get_config()["batch_size"], shuffle=True
)

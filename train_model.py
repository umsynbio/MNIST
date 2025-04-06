import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from training.train import train_one_epoch, evaluate
from config.config import get_config
from dataset.make_dataset import train_loader, test_loader

from models.model import SimpleNN


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    epochs = config["epochs"]
    num_classes = config.get("num_classes", 10)  # fallback to 10 for MNIST

    run = wandb.init(
        project="mnist-basic",
        config={
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "architecture": "CNN",
            "dataset": "MNIST",
            "epochs": epochs,
        },
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, pr_auc = evaluate(
            model, test_loader, criterion, device, num_classes
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Accuracy: {val_acc:.2f}% "
            f"PR-AUC: {pr_auc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_pr_auc": pr_auc,
            }
        )


    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("mnist-basic", type="model")
    artifact.add_file("model.pth")
    run.log_artifact(artifact)
    
    run.finish()


if __name__ == "__main__":
    main()

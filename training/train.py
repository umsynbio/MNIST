import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from models.model import SimpleNN
from config.config import get_config
from dataset.make_dataset import train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            eval_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_targets_bin = label_binarize(all_targets, classes=np.arange(num_classes))

    pr_auc = average_precision_score(all_targets_bin, all_probs, average="macro")
    accuracy = 100 * correct / total
    avg_eval_loss = eval_loss / len(loader)

    return avg_eval_loss, accuracy, pr_auc
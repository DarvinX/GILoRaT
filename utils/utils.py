import torch
import csv
import os

def evaluate(model, loader, criterion, device="cuda:0", isBaseline=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item() * x.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(y).sum().item()
            total += x.size(0)
    acc = correct / total
    model.train()

    return total_loss / total, acc


class Logger:
    def __init__(self, filename, dir="./logs", cols=['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Trainable Params', 'Test Accuracy']):
        self.filepath = os.path.join(dir, filename)
        self.cols = cols
        with open(self.filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(cols)

    def log(self, data):
        assert len(data) == len(self.cols), f"expected columns {self.cols} but got {data}"
        with open(self.filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)


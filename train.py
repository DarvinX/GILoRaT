import argparse
from models.AlexNet import AlexNet
from models.LeNet5 import LeNet5
from utils.utils import Logger, evaluate
from core.og_wrapper import change_to_dlora, custom_optim_params, ortho_loss_fn
from dataloader.CIFAR10 import CIFAR10Dataloader
from dataloader.MNIST import MNISTDataloder

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--init_rank", type=int, default=2)
parser.add_argument("--increment", type=int, default=2)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--print_batch", type=int, default=300)
parser.add_argument("--model", type=str, choices=["LeNet", "AlexNet"])
parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10"])
parser.add_argument("--use_ortho_loss", action=argparse.BooleanOptionalAction)
parser.add_argument("--log_name", type=str)
parser.add_argument("--lr", type=float, default=0.001)

args = parser.parse_args()

if __name__ == "__main__":

    print(args)
    # exit()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # hyperparameters
    init_rank = args.init_rank
    increment = args.increment
    epochs = args.epochs
    print_batch = args.print_batch

    net = AlexNet() if args.model == "AlexNet" else LeNet5()
    
    r = 1
    first = True
    for i in range(init_rank):        
        change_to_dlora(net, first=first)
        first = False
        r+=1

    criterion = nn.CrossEntropyLoss()

    param_list = custom_optim_params(net)
    optimizer = optim.Adam(param_list, lr=0.001)

    dataLoader = CIFAR10Dataloader() if args.dataset == "CIFAR10" else MNISTDataloder()
    
    if not args.log_name:
        raise Exception("Please specify the name of the log file")

    logger = Logger(args.log_name, cols=["epoch",
                                          "step", 
                                          "train_loss", 
                                          "train_acc", 
                                          "val_loss", 
                                          "val_acc", 
                                          "num_params", 
                                          "test_acc",
                                          "rank"])
    
    prev_running_loss = 0.0
    best_acc = 0
    wait = 0
    val_acc_history = []

    net = net.to(device)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        d_progress = tqdm(dataLoader.train_loader)
        for i, data in enumerate(d_progress,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            if args.use_ortho_loss:
                ortho_loss = ortho_loss_fn(net)
                loss += ortho_loss

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_batch == (print_batch-1):    # print every 2000 mini-batches
                running_loss /= print_batch
                val_loss, val_acc = evaluate(net, dataLoader.val_loader, criterion, device)
                test_loss, test_acc = evaluate(net, dataLoader.test_loader, criterion, device)
                train_loss, train_acc = evaluate(net, dataLoader.train_loader, criterion, device)
                val_acc_history.append(val_acc)
                net.train()

                if val_acc > best_acc:
                    wait = 0
                    best_acc = val_acc
                else:
                    wait += 1

                num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test_Acc : {test_acc:.4f}")
                logger.log([epoch+1, i,  train_loss, train_acc, val_loss, val_acc, num_params, test_acc, r-1])

                if wait >= r*2: #patience: consider patience as r
                    # print("saved")
                    torch.save(net.state_dict(), F'./cifar_net_r{r-1}.pth')
                    wait = 0

                    for _ in range(increment):
                        change_to_dlora(net, r=r)
                        r+=1
                    # print(f'ortho_loss: {ortho_loss}')
                    net.to("cuda:0")
                    summary(net, (3,32,32))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(custom_optim_params(net), lr=0.001)

                    running_loss = 0.0
                prev_running_loss = running_loss
                running_loss = 0.0


    print('Finished Training')



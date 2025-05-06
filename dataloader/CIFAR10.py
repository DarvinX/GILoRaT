from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader, random_split

class CIFAR10Dataloader:
    def __init__(self, train_transform=T.Compose([
            # T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
            ]), 
            test_transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010))]),
            root="./data", batch_size=64, num_workers=2, split=(0.8,0.2)):

        full_trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
        
        trainset, valset = random_split(full_trainset, split)

        self.train_loader = DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
        
        self.val_loader = DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
        
        testset = datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
        self.test_loader = DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
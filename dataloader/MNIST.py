from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MNISTDataloder:
    def __init__(self, batch_size=256, 
                 train_transform = [transforms.ToTensor()], 
                 test_transform=[transforms.ToTensor()], 
                 shuffle=(True, False, False),
                 split=[50000, 10000],
                 root="./data"):
        if isinstance(batch_size, int):
            batch_size = (batch_size, batch_size, batch_size)

        
        # transform = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.Compose(train_transform))
        test_set = datasets.MNIST(root=root, train=False, download=True, transform=transforms.Compose(test_transform))
        train_set, val_set = random_split(full_train, split)
        self.train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=shuffle[0])
        self.val_loader = DataLoader(val_set, batch_size=batch_size[1], shuffle=shuffle[1])
        self.test_loader = DataLoader(test_set, batch_size=batch_size[2], shuffle=shuffle[2])

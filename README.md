# GILoRaT
Greedy Incremental Low Rank Training

### Example
```bash
python train.py --model AlexNet --epochs 30 --dataset CIFAR10 --use_ortho_loss --log_name AlexNet_CIFAR10_ortho_aflora.csv
```

run the following command to get all the arguments
```bash
python train.py --help
```

### supported models
* AlexNet
* LeNet5

### datasets
* MNIST
* CIFAR10
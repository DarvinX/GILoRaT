import torch.nn as nn
from layers import GILoRaTLinear, GILoRaTConv2dMatrix

class GILoRaTWrapper(nn.Module):
    def __init__(self, model, epsilon=1e-3, input_size=(1, 28, 28), init_rank = 2, increment = 2):
        super().__init__()
        self.model = self._wrap(model)
        self.patience = init_rank * 2
        self.epsilon = epsilon
        self.input_size = input_size
        self.val_acc_history = []
        self.optimizer = None
        self.rank = init_rank
        self.increment = increment

    def _wrap(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, GILoRaTLinear(module.in_features, module.out_features))
            # elif isinstance(module, nn.Conv2d):
            #     setattr(model, name, GILoRaTConv2dMatrix(
            #         module.in_channels,
            #         module.out_channels,
            #         module.kernel_size,
            #         stride=module.stride,
            #         padding=module.padding
            #     ))
            else:
                self._wrap(module)
        return model

    def forward(self, x):
        return self.model(x)

    def update_validation_accuracy(self, val_acc):
        self.val_acc_history.append(val_acc)
        if len(self.val_acc_history) > self.patience:
            delta = max(self.val_acc_history[-self.patience:]) - self.val_acc_history[-self.patience]
            if delta < self.epsilon:
                self.increase_all_ranks()
                self.val_acc_history = []

    def increase_all_ranks(self):
        print("[GILoRaTWrapper] Stagnation detected. Increasing rank...")
        new_params = []
        self.rank += self.increment
        self.patience = self.rank * 2
        for module in self.model.modules():
            if isinstance(module, GILoRaTLinear) or isinstance(module, GILoRaTConv2dMatrix):
                old_params = set(module.parameters())
                module.increase_rank(self.increment)
                new_params.extend([p for p in module.parameters() if p not in old_params])
    # Add new parameters to optimizer
        if self.optimizer:
            self.optimizer.add_param_group({'params': new_params}, lr = 0.001)
    #self.print_summary()


    # def print_summary(self):
    #     device = next(self.parameters()).device
    #     print("\n[Model Summary after Rank Increment]")
    #     summary(self.model.to(device), input_size=self.input_size)
    #     total_params = sum(p.numel() for p in self.model.parameters())
    #     total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
    #     print(f"Total Params: {total_params}, Estimated Size: {total_bytes / (1024**2):.2f} MB")
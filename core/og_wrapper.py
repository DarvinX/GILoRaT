import torch
import torch.nn as nn

class DLoRA(torch.nn.Module):
    def __init__(self, old_layer, r=1, first=False):
        super().__init__()
        self.in_features = old_layer.in_features
        self.out_features = old_layer.out_features
        og_r = r
        r = min(r, min(self.in_features, self.out_features)//3)
        self.a = nn.Linear(self.in_features, r, bias=False)
        self.b = nn.Linear(r, self.out_features)


        # if first:
        #     self.linear = old_layer
        # else:
        #     self.linear = old_layer.linear
        # # print(self.old_layer.weight.shape)


        # if first:
        #     self.linear.weight = nn.Parameter(torch.zeros_like(old_layer.weight))
        #     #  torch.empty_like(self.old_layer.weight)
        if not first:

            nn.init.normal_(self.b.weight, 0, 1)
            # nn.init.normal_(self.b.weight, 0, 1)
            self.a.weight = nn.Parameter(torch.zeros_like(self.a.weight))

            # print(self.a.weight[1,2:7])
            if r > 1:
                with torch.no_grad():
                    # self.linear.weight += nn.Parameter(torch.matmul(old_layer.b.weight, old_layer.a.weight))
                    if og_r == r:
                        self.a.weight[:-1,:] = old_layer.a.weight
                        self.b.weight[:,:-1] = old_layer.b.weight
                    else:
                        self.a.weight = old_layer.a.weight
                        self.b.weight = old_layer.b.weight
                    # adapt for models with no bias
                    self.b.bias = old_layer.b.bias


        # self.linear.weight.requires_grad = False

        # print(self.linear.weight.shape)
        # print(self.a.weight.shape)
        # print(self.b.weight.shape)


    def forward(self, x):
        # y = self.linear(x)

        x = self.a(x)
        x = self.b(x)

        return x
    
def custom_optim_params(net, lr_a=0.001, lr_b=0.01):
  params_list = []

  for name, module in net.named_children():
    # print(name)
    if isinstance(module, DLoRA):
      params_list.append({"params": module.a.parameters(), "lr": lr_a})
      params_list.append({"params": module.b.parameters(), "lr": lr_b})
    else:
      # check if module contains any parameters
      if len(list(module.parameters())) > 0:
        params_list.append({"params": module.parameters()})
  return params_list


def loss_function(matrix):
  sum = 0
  if matrix.size(0) <= 1:
    return 0

  for i in range(1, matrix.size(0)):
    for j in range(i):
      sum += torch.abs(torch.dot(matrix[j], matrix[i]))
  return sum


def ortho_loss_fn(net):
  loss = 0
  for name, module in net.named_modules():
        if isinstance(module, DLoRA):
            loss += loss_function(module.a.weight)
  return loss


def change_to_dlora(net, r=1, first=False):
  last_layer_name = list(net.named_modules())[-1][0]
#   print(last_layer_name)
  changes_to_make = []

  for name, module in net.named_modules():
    # print(name)

    if isinstance(module, nn.Linear if first else DLoRA):
      # print(module.weight.shape)
      # print(module)
      if name == last_layer_name:
        continue

      if first:
        size = module.weight.shape
        total_params = size[0]*size[1]

        if total_params > 2000:
        #   print("appending")
          dlora_layer = DLoRA(module, first=first)
          changes_to_make.append((name, dlora_layer))

      else:
        # print("not the first pass")
        dlora_layer = DLoRA(module, first=first, r=r)
        # setattr(net, name, dlora_layer)
        # print("appending")
        changes_to_make.append((name, dlora_layer))

  for name, module in changes_to_make:
    # print("changing")
    # print(name)
    # print(module
    if "." in name:
      # parent_module, name_in_parent = name.rsplit('.', 1)
    #   print("parent_keys",dict(net.named_modules()).keys())
    #   print(name)

      *parent_name, submodule_name = name.split('.')
      parent = net
      for part in parent_name:
          parent = getattr(parent, part)

      # parent = dict(module.named_modules())[name]

      setattr(parent, submodule_name, module)
    else:
      setattr(net, name, module)
import torch
import torch.nn as nn

class DLoRA(torch.nn.Module):
    def __init__(self, old_layer, r=1, first=False, rank_limit_factor=3, ignore_score=True):
        super().__init__()
        self.in_features = old_layer.in_features
        self.out_features = old_layer.out_features
        og_r = r
        self.rank_limit = min(self.in_features, self.out_features)//rank_limit_factor
        self.r = min(r, self.rank_limit)
        self.a = nn.Linear(self.in_features, self.r, bias=False)
        self.b = nn.Linear(self.r, self.out_features)

        if not first:
            
            nn.init.normal_(self.b.weight, 0, 1)
            # nn.init.normal_(self.b.weight, 0, 1)
            self.a.weight = nn.Parameter(torch.zeros_like(self.a.weight))
            if r > 1:
                with torch.no_grad():
                    if og_r == self.r:
                        self.a.weight[:-1,:] = old_layer.a.weight
                        self.b.weight[:,:-1] = old_layer.b.weight
                    else:
                        self.a.weight = old_layer.a.weight
                        self.b.weight = old_layer.b.weight
                    # adapt for models with no bias
                    self.b.bias = old_layer.b.bias

        self.tracker_A = Inc_ScoreTracker(self.a.weight.shape, device=self.a.weight.device)
        # self.tracker_B = Inc_ScoreTracker(self.b.weight.shape, device=self.b.weight.device)

        self.a.weight.register_hook(
            lambda g: self._ema_update_tracker(self.tracker_A, g))
        # self.b.weight.register_hook(
        #     lambda g: self._ema_update_tracker(self.tracker_B, g))
        
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)

        return x
    
    def _ema_update_tracker(self, tracker, grad, beta: float = 0.9):
        tracker.update(grad.detach().clone())
    
    # def ema_grad_update(self):
    #    with torch.no_grad():
    #       if self.a.grad is not None:
    #           self.tracker_A.update(self.a.grad.detach().clone())
    #       if self.b.grad is not None:
    #           self.tracker_B.update(self.b.grad.detach().clone())

    # def _post_backward_hook(self, module, grad_input, grad_output):
    #     self.ema_grad_update()
    
    def get_score(self):
      #  score_A = self.tracker_A.compute_score()
      #  score_B = self.tracker_B.compute_score()
      #  combined_score = (score_A + score_B) / 2
       return self.tracker_A.compute_score()

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


def change_to_dlora(net, r=1, first=False, threshold=0.001, ignore_score=True):
  last_layer_name = list(net.named_modules())[-1][0]
  changes_to_make = []

  for name, module in net.named_modules():

    if isinstance(module, nn.Linear if first else DLoRA):
      if name == last_layer_name:
        continue

      if first:
        size = module.weight.shape
        total_params = size[0]*size[1]

        if total_params > 2000:
          dlora_layer = DLoRA(module, first=first)
          changes_to_make.append((name, dlora_layer))

      elif ignore_score:
        dlora_layer = DLoRA(module, first=first, r=module.r+1)
        changes_to_make.append((name, dlora_layer))
      else:
        print("layer score", module.get_score())

        if module.get_score() < threshold:
           print("pre rank", module.r)
           dlora_layer = DLoRA(module, first=first, r=module.r+1)
           changes_to_make.append((name, dlora_layer))
           print("post_rank rank:", dlora_layer.r)
        

  for name, module in changes_to_make:
    if "." in name:
      *parent_name, submodule_name = name.split('.')
      parent = net
      for part in parent_name:
          parent = getattr(parent, part)

      setattr(parent, submodule_name, module)
    else:
      setattr(net, name, module)

class Inc_ScoreTracker:
    def __init__(self, shape, beta=0.9, device='cpu'):
        self.beta = beta
        self.eps = 1e-8 

        self.ema_grad = torch.zeros(shape, device=device)
        self.ema_grad_sq = torch.zeros(shape, device=device)
        self.step = 0

    def update(self, grad):
        self.step += 1
        self.ema_grad = self.ema_grad.to(grad.device)
        self.ema_grad = self.beta * self.ema_grad + (1 - self.beta) * torch.abs(grad)
        self.ema_grad_sq = self.ema_grad_sq.to(grad.device)
        self.ema_grad_sq = self.beta * self.ema_grad_sq + (1 - self.beta) * grad**2

    def compute_score(self):
        var = self.ema_grad_sq - self.ema_grad**2
        var = torch.clamp(var, min=self.eps)
        uncertainty = torch.sqrt(var)

        inc_score = torch.mean(self.ema_grad * uncertainty)
        return inc_score.item()
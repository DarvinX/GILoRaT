import torch
import torch.nn as nn

# linear
class GILoRaTLinear(nn.Module):
    def __init__(self, in_features, out_features, initial_rank=2, max_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = initial_rank
        self.max_rank = max_rank or (min(in_features, out_features) //3)

        self.A = nn.Parameter(torch.empty(out_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.A, mode='fan_out')
        nn.init.kaiming_normal_(self.B, mode='fan_in')

    def forward(self, x):
        x = F.linear(x, self.B)
        x = F.linear(x, self.A)
        return x

    def _orthogonal_kaiming_extension(self):
        A_tilde = torch.empty(self.out_features, 1, device=self.A.device)
        B_tilde = torch.empty(1, self.in_features, device=self.B.device)
        nn.init.kaiming_normal_(A_tilde, mode='fan_out')
        nn.init.zeros_(B_tilde)

        # dot = torch.matmul(A_tilde, B_tilde)
        # norm_sq = A_tilde.norm() ** 2 + 1e-8
        # B_tilde -= (dot / norm_sq).squeeze() * A_tilde.T.squeeze()

        return A_tilde, B_tilde

    def increase_rank(self, increment=2):
        if self.rank + increment > self.max_rank:
            return
        A_exts, B_exts = [], []
        for _ in range(increment):
            A_tilde, B_tilde = self._orthogonal_kaiming_extension()
            A_exts.append(A_tilde)
            B_exts.append(B_tilde)
        self.A = nn.Parameter(torch.cat([self.A.data] + A_exts, dim=1))
        self.B = nn.Parameter(torch.cat([self.B.data] + B_exts, dim=0))
        self.rank += increment
        print(f"[GILoRaTLinear] Increased rank to {self.rank}")


# CNN
class GILoRaTConv2dMatrix(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initial_rank=2, max_rank=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        # Ensure padding is an integer or a tuple of two integers
        if isinstance(padding, int):
            self.padding = (padding, padding)  # Convert to tuple if it's an integer
        elif isinstance(padding, tuple) and len(padding) == 2:
            self.padding = padding
        elif isinstance(stride, int):
            self.stride = (stride, stride)  # Convert to tuple if it's an integer
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise ValueError("padding must be an integer or a tuple of two integers")
        self.rank = initial_rank
        self.max_rank = max_rank or ((in_channels * self.kernel_size[0] * self.kernel_size[1])//3)

        # Compute flattened kernel size
        self.kernel_flat_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]

        # Low-rank matrices A and B
        self.A = nn.Parameter(torch.empty(out_channels, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, self.kernel_flat_dim))

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.A, mode='fan_out')
        nn.init.kaiming_normal_(self.B, mode='fan_in')

    def forward(self, x):
        batch_size = x.size(0)
        patches = self.unfold(x)  # [N, CJK, L]
        out = self.A @ (self.B @ patches)  # [F, L]
        out = out.view(batch_size, self.out_channels, -1)

        # Compute output H and W
        H_out = (x.size(2) + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1 # Access first element of tuple
        W_out = (x.size(3) + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1 # Access second element of tuple

        return out.view(batch_size, self.out_channels, H_out, W_out)

    def _orthogonal_kaiming_extension(self):
        A_tilde = torch.empty(self.out_channels, 1, device=self.A.device)
        B_tilde = torch.empty(1, self.kernel_flat_dim, device=self.B.device)
        nn.init.kaiming_normal_(A_tilde, mode='fan_out')
        nn.init.zeros_(B_tilde)

        dot = torch.matmul(A_tilde, B_tilde)
        norm_sq = A_tilde.norm() ** 2 + 1e-8
        B_tilde -= (dot / norm_sq) * A_tilde.T

        return A_tilde, B_tilde

    def increase_rank(self, increment=1):
        if self.rank + increment > self.max_rank:
            print("[GILoRaTConv2d] Max rank reached.")
            return

        new_As, new_Bs = [], []
        for _ in range(increment):
            A_tilde, B_tilde = self._orthogonal_kaiming_extension()
            new_As.append(A_tilde)
            new_Bs.append(B_tilde)

        self.A = nn.Parameter(torch.cat([self.A.data] + new_As, dim=1))
        self.B = nn.Parameter(torch.cat([self.B.data] + new_Bs, dim=0))
        self.rank += increment
        print(f"[GILoRaTConv2d] Rank increased to {self.rank}")
import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, u_weight, v_weight, bias=None):
        super().__init__()
        rank = 0 if u_weight.numel() == 0 else u_weight.shape[1]
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=bias is not None)

        self.v_proj.weight.data.copy_(v_weight)
        self.u_proj.weight.data.copy_(u_weight)
        if bias is not None:
            self.u_proj.bias.data.copy_(bias)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))


class ZeroLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.to(device=device, dtype=dtype).clone(), requires_grad=False)

    def forward(self, x):
        output_shape = x.shape[:-1] + (self.out_features,)
        out = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

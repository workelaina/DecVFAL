import torch
from torch import nn, Tensor

from utils import split_x

from .constant import *
from .models import my_mlp, my_resnet


class Vfl(nn.Module):
    def __init__(
        self,
        n_client: int,
        model_name: str,
        model_args: list
    ) -> None:
        assert 1 <= n_client
        super().__init__()
        self.n_client = n_client
        self.model_name = model_name
        self.model_args = model_args

        self.server = nn.Linear(model_args[-2]*n_client, model_args[-1])
        # self.server = nn.Sequential(
        #     # nn.Linear(model_args[-2]*n_client, model_args[-1])
        #     nn.Linear(model_args[-2]*n_client, 256),
        #     nn.ReLU(True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(True),
        #     nn.Linear(256, model_args[-1])
        # )
        if model_name == 'mlp':
            f = my_mlp
        elif model_name == 'resnet':
            f = my_resnet
        else:
            raise ValueError(model_name)
        self.clients = nn.Sequential(*[
            f(model_args[:-1]) for _ in range(n_client)
        ])

    def _x2msg(self, x: Tensor) -> Tensor:
        xs = split_x(x, self.n_client, True)
        ms = list()
        for i in range(self.n_client):
            ms.append(self.clients[i](xs[i]))
        return torch.cat(ms, dim=-1)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        m = self._x2msg(x)
        y = self.server(m)
        return y

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @staticmethod
    def load_vfl(pth: str) -> nn.Module:
        pt = torch.load(pth)
        model = Vfl(
            pt[VFL_N_CLIENT],
            pt[VFL_MODEL_NAME],
            pt[VFL_MODEL_ARGS]
        )
        model.load_state_dict(pt[VFL_MODEL], strict=True)
        return model

    def save_vfl(self, pth: str) -> None:
        torch.save({
            VFL_N_CLIENT: self.n_client,
            VFL_MODEL_NAME: self.model_name,
            VFL_MODEL_ARGS: self.model_args,
            VFL_MODEL: self.state_dict()
        }, pth)

from typing import Literal

import torch
from torch import nn, Tensor
from torch.nn import Sequential

from .constant import *
from .abclass import Lw_1


class Lw_layer(Lw_1):
    def __init__(
        self,
        name: str,
        device: str,
        up: Lw_1,
        model: Sequential,
        lr: float,
        up_n: int = 0,
        n_down: int = 1,
        mp_flg: Literal['async', 'sync', None] = 'sync',
        count_delta: int = None
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [999],
        )
        super().__init__(name, device, up, up_n, n_down, mp_flg, count_delta)

    def act_fwd(self, data: dict) -> None:
        assert DATA_LEVEL in data
        _grad_lv = data[DATA_LEVEL]
        count = data[DATA_COUNT]
        x: Tensor = data[DATA_X]
        x = x.to(self.device)

        if _grad_lv <= GRAD_FREE:
            self.prt(15, count, 'with torch.no_grad():')
            with torch.no_grad():
                y: Tensor = self.model(x)
                self.put_to_up_data(data, y)
            return

        x.requires_grad_()
        x.retain_grad()

        y: Tensor = self.model(x)
        self.put_to_up_data(data, y.detach())
        y_grad = self.get_from_up_grad(count)

        x_grad = torch.autograd.grad(
            y,
            x,
            grad_outputs=y_grad,
            only_inputs=True,
            retain_graph=_grad_lv >= GRAD_BWD,
            allow_unused=False
        )[0]
        assert x_grad is not None
        self.prt(20, count, 'layer.put_to_down_cache(x_grad)')
        self.put_to_down_cache(count, x_grad)

        if _grad_lv >= GRAD_BWD:
            y.backward(
                gradient=y_grad,
                retain_graph=False,
                inputs=list(self.model.parameters())
            )

    def zg(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def _end_epoch(self) -> None:
        self.lr_scheduler.step()


class Lw_layer_inner(Lw_layer):
    def __init__(
        self,
        name: str,
        device: str,
        up: Lw_1,
        model: Sequential,
        lr: float,
        up_n: int = 0,
        mp_flg: bool = True,
        count_delta: int = None
    ) -> None:
        super().__init__(
            name, device, up, model, lr,
            up_n, 1,
            mp_flg, count_delta
        )

    def act_ours(self, data: dict) -> None:
        assert DATA_LEVEL in data
        _grad_lv = data[DATA_LEVEL]
        assert _grad_lv >= GRAD_ONLY_X
        count = data[DATA_COUNT]
        x: Tensor = data[DATA_X]
        x = x.to(self.device)

        x.requires_grad_()
        x.retain_grad()

        y: Tensor = self.model(x)
        y_grad = self.get_from_up_grad(count)
        h = torch.sum(y*y_grad)

        x_grad = torch.autograd.grad(
            h,
            x,
            only_inputs=True,
            retain_graph=_grad_lv >= GRAD_BWD,
            allow_unused=False
        )[0]
        assert x_grad is not None
        self.prt(20, count, 'l_i.put_to_down_cache(inner_grad)')
        self.put_to_down_cache(count, x_grad)

        if _grad_lv >= GRAD_BWD:
            h.backward(
                retain_graph=False,
                inputs=list(self.model.parameters())
            )

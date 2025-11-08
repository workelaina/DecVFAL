from typing import Literal

import torch
from torch import nn, Tensor
from torch.nn import Sequential

from com import Compressor
from zo import Zor
from utils import Result

from .constant import *
from .abclass import Lw_0


class Lw_top(Lw_0):
    def __init__(
        self,
        name: str,
        device: str,
        res: Result,
        n_down: int = 1,
        mp_flg: Literal['async', 'sync', None] = None
    ) -> None:
        self.res = res
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        super().__init__(name, device, n_down, mp_flg, 0)

    def _end_epoch(self) -> None:
        _s = '%s: Epoch: %d, Loss: %.5f, Acc: %.3f%%, %.3fMB/%.3fs'
        print(_s % self.res.end_epoch())

    def act_fwd(self, data: dict) -> None:
        assert DATA_LEVEL in data
        _grad_lv = data[DATA_LEVEL]
        count = data[DATA_COUNT]
        y: Tensor = data[DATA_X]
        y = y.to(self.device)
        labels: Tensor = data[DATA_LABEL]
        if labels is None:
            labels = self.labels_cache
            append_log = False
        else:
            labels = labels.to(self.device)
            self.labels_cache = labels
            append_log = True

        if _grad_lv <= GRAD_FREE:
            self.prt(15, count, 'with torch.no_grad():')
            with torch.no_grad():
                loss = self.loss_fn(y, labels)
                if append_log:
                    self.res.batch(y, labels, loss)
            return

        y.requires_grad_()
        y.retain_grad()

        loss = self.loss_fn(y, labels)
        if append_log:
            self.res.batch(y, labels, loss)

        y_grad = torch.autograd.grad(
            loss,
            y,
            only_inputs=True,
            retain_graph=False,
            allow_unused=False
        )[0]
        assert y_grad is not None
        self.prt(20, count, 'top.put_to_down_cache(y_grad)')
        self.put_to_down_cache(count, y_grad)


class Lw_layer_top(Lw_top):
    def __init__(
        self,
        name: str,
        device: str,
        model: Sequential,
        lr: float,
        res: Result,
        n_down: int,
        mp_flg: Literal['async', 'sync', None] = 'sync'
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [999],
        )
        super().__init__(name, device, res, n_down, mp_flg)

    def zg(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def _end_epoch(self) -> None:
        self.lr_scheduler.step()
        super()._end_epoch()

    def act_fwd(self, data: dict) -> None:
        assert DATA_LEVEL in data
        _grad_lv = data[DATA_LEVEL]
        count = data[DATA_COUNT]
        x: Tensor = data[DATA_X]
        x = x.to(self.device)
        labels: Tensor = data[DATA_LABEL]
        if labels is None:
            labels = self.labels_cache
            append_log = False
        else:
            labels = labels.to(self.device)
            self.labels_cache = labels
            append_log = True

        if _grad_lv <= GRAD_FREE:
            self.prt(15, count, 'with torch.no_grad():')
            with torch.no_grad():
                y: Tensor = self.model(x, data)
                loss: Tensor = self.loss_fn(y, labels)
                if append_log:
                    self.res.batch(y, labels, loss)
            return

        x.requires_grad_()
        x.retain_grad()

        y: Tensor = self.model(x)
        loss: Tensor = self.loss_fn(y, labels)
        if append_log:
            self.res.batch(y, labels, loss)

        x_grad = torch.autograd.grad(
            loss,
            x,
            only_inputs=True,
            retain_graph=_grad_lv >= GRAD_BWD,
            allow_unused=False
        )[0]
        assert x_grad is not None
        self.prt(20, count, 'l_top.put_to_down_cache(x_grad)')
        self.put_to_down_cache(count, x_grad)

        if _grad_lv >= GRAD_BWD:
            loss.backward(
                retain_graph=False,
                inputs=list(self.model.parameters())
            )


class Lw_czo_top(Lw_layer_top):
    def __init__(
        self,
        name: str,
        device: str,
        model: Sequential,
        lr: float,
        res: Result,
        compressor: Compressor,
        zor: Zor,
        n_down: int,
        mp_flg: Literal['async', 'sync', None] = 'sync'
    ) -> None:
        self.compressor = compressor
        self.zor = zor
        super().__init__(name, device, model, lr, res, n_down, mp_flg)

    def act_fwd(self, data: dict) -> None:
        assert DATA_LEVEL in data
        _grad_lv = data[DATA_LEVEL]
        count = data[DATA_COUNT]
        x: Tensor = data[DATA_X]
        x = x.to(self.device)
        labels: Tensor = data[DATA_LABEL]
        if labels is None:
            labels = self.labels_cache
            append_log = False
        else:
            labels = labels.to(self.device)
            self.labels_cache = labels
            append_log = True

        x_e, msg_size = self.compressor.c_dec(x)
        self.res.communicate(msg_size)

        if _grad_lv <= GRAD_FREE:
            with torch.no_grad():
                y: Tensor = self.model(x_e, data)
                loss: Tensor = self.loss_fn(y, labels)
                if append_log:
                    self.res.batch(y, labels, loss)
            return

        if self.zor is None:
            x_e.requires_grad_()
            x_e.retain_grad()

        y: Tensor = self.model(x_e)
        loss: Tensor = self.loss_fn(y, labels)
        if append_log:
            self.res.batch(y, labels, loss)

        if self.zor is None:
            x_grad = torch.autograd.grad(
                loss,
                x_e,
                only_inputs=True,
                retain_graph=_grad_lv >= GRAD_BWD,
                allow_unused=False
            )[0]
            msg_size = x_grad.element_size() * 8
        else:
            def f(_x: Tensor) -> Tensor:
                return self.loss_fn(self.model(_x), labels)
            x_grad = self.zor.get_x_grad(x_e, f, loss)
            msg_size = self.zor.delta_l * loss.element_size() * 8

        self.res.r_communicate(msg_size)

        assert x_grad is not None
        self.prt(20, count, 'czo_top.put_to_down_cache(x_grad)')
        self.put_to_down_cache(count, x_grad)

        if _grad_lv >= GRAD_BWD:
            loss.backward(
                retain_graph=False,
                inputs=list(self.model.parameters())
            )

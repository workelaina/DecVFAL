import os
import sys
from typing import Tuple, Iterable

import torch
from torch import nn, Tensor

from utils import mask_x
from zo import Zor
from model import Vfl

from .cps import Cps_all

class Atk_clean:
    def __init__(
        self,
        cps: Cps_all = None,
        name: str = 'clean'
    ) -> None:
        self.loss_fn = nn.CrossEntropyLoss()
        self.cps = cps
        self.name = name

    def masked(self, x: Tensor) -> Tensor:
        if self.cps is not None:
            _is, _n = self.cps.gt()
            x = mask_x(x, _is, _n)
        return x

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        return x

    def atk(self, model: nn.Module, x: Tensor, labels: Tensor) -> None:
        adv_x = self.gen_atk_x(model, x, labels)
        with torch.no_grad():
            y = model(adv_x)
            loss = self.loss_fn(y, labels)
        if self.cps is not None:
            self.cps.up(y, labels)
        return y, labels, loss


class Atk_cer(Atk_clean):
    def __init__(
        self,
        epsilon: float,
        eta_mask: Tuple[Iterable[int], int] = None,
        name: str = 'cer'
    ) -> None:
        self.epsilon = epsilon
        super().__init__(eta_mask, name)

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        eta = torch.randn_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = x + self.masked(eta)
        adv_x.clamp_(0., 1.)
        return adv_x


class Atk_fgsm(Atk_clean):
    def __init__(
        self,
        epsilon: float,
        zor: Zor,
        eta_mask: Tuple[Iterable[int], int] = None,
        name: str = 'fgsm'
    ) -> None:
        self.epsilon = epsilon
        self.zor = zor
        super().__init__(eta_mask, name)

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        def f(_x: Tensor) -> Tensor:
            y = model(_x)
            loss = self.loss_fn(y, labels)
            return loss

        if self.zor is None:
            x.requires_grad_()
            x.retain_grad()
            loss = f(x)
            x_grad = torch.autograd.grad(
                loss, x,
                only_inputs=True,
                retain_graph=False,
                allow_unused=False
            )[0].detach()
        else:
            x_grad = self.zor.get_x_grad(x, f)

        adv_x = x + self.masked(x_grad.sign() * self.epsilon)
        adv_x.clamp_(0., 1.)
        return adv_x


class Atk_pgd(Atk_clean):
    def __init__(
        self,
        n: int,
        epsilon: float,
        sigma: float,
        rand_start: bool,
        zor: Zor,
        eta_mask: Tuple[Iterable[int], int] = None,
        name: str = 'pgd'
    ) -> None:
        self.n = n
        self.epsilon = epsilon
        self.sigma = sigma
        self.rand_start = rand_start
        self.zor = zor
        super().__init__(eta_mask, name)

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        if self.rand_start:
            eta = torch.randn_like(x).uniform_(-self.epsilon, self.epsilon)
            eta = self.masked(eta)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        def f(_x: Tensor) -> Tensor:
            y = model(_x)
            loss = self.loss_fn(y, labels)
            return loss

        for _i in range(self.n):
            if self.zor is None:
                adv_x.requires_grad_()
                adv_x.retain_grad()
                loss = f(adv_x)
                x_grad = torch.autograd.grad(
                    loss, adv_x,
                    only_inputs=True,
                    retain_graph=False,
                    allow_unused=False
                )[0].detach()
            else:
                x_grad = self.zor.get_x_grad(adv_x, f)

            eta += self.masked(x_grad.sign() * self.sigma)
            eta.clamp_(-self.epsilon, self.epsilon)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)

        return adv_x


class Atk_aa(Atk_clean):
    def __init__(
        self,
        n: int,
        m: int,
        y_size: int,
        epsilon: float,
        resc_schedule: bool,
        eta_mask: Tuple[Iterable[int], int] = None,
        name: str = 'aa'
    ) -> None:
        self.n = n
        self.m = m
        self.y_size = y_size
        self.epsilon = epsilon
        self.resc_schedule = resc_schedule
        super().__init__(eta_mask, name)

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        from autoattack import AutoAttack
        adversary = AutoAttack(
            model,
            eps=self.epsilon,
            verbose=False,
            device=x.device,
            n_iter=self.n,
            san=self.m,
            sa_re=self.resc_schedule,
            outputs_size=self.y_size
        )

        adv_x: Tensor = adversary.run_standard_evaluation(
            x, labels, bs=x.shape[0]
        )
        eta = self.masked(adv_x - x)
        eta.clamp_(-self.epsilon, self.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        return adv_x


class Atk_cw(Atk_clean):
    def __init__(
        self,
        n: int,
        m_iter_check: int,
        y_size: int,
        sigma: float,
        c: float,
        prev: float,
        use_tanh: bool,
        zor: Zor,
        eta_mask: Tuple[Iterable[int], int] = None,
        name: str = 'cw'
    ) -> None:
        self.n = n
        self.m_iter_check = m_iter_check
        self.y_size = y_size
        self.sigma = sigma
        self.c = c
        self.prev = prev
        self.use_tanh = use_tanh
        self.zor = zor
        assert not use_tanh
        super().__init__(eta_mask, name)

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        adv_x = x + self.masked(.5 - x)
        prev = self.prev

        tanh = nn.Tanh()
        l2_fn = nn.MSELoss(reduction='sum')

        def f(_x: Tensor) -> Tensor:
            l2 = l2_fn(_x, x)
            y = model(_x)
            loss = self.loss_fn(y, labels)
            return self.c * loss - l2

        for _i in range(self.n):
            if self.zor is None:
                adv_x.requires_grad_()
                adv_x.retain_grad()
                loss = f(adv_x)
                x_grad = torch.autograd.grad(
                    loss, adv_x,
                    only_inputs=True,
                    retain_graph=False,
                    allow_unused=False
                )[0].detach()
                adv_x = adv_x.detach()
            else:
                x_grad = self.zor.get_x_grad(adv_x, f)

            adv_x += self.masked(x_grad * self.sigma)
            adv_x.clamp_(0., 1.)

            if _i % self.m_iter_check == 0:
                if loss > prev:
                    print('cw: Attack Stopped due to CONVERGENCE')
                    break
                prev = loss.detach()
            # if self.use_tanh:
            #     eta = (tanh(adv_x) + 1) / 2 - x
            #     adv_x = x + self.masked(eta)
            #     adv_x.clamp_(0., 1.)

        return adv_x


class Atk_empgd(Atk_pgd):
    def __init__(
        self,
        n: int,
        epsilon: float,
        sigma: float,
        rand_start: bool,
        zor: Zor,
        eta_mask: Tuple[Iterable[int] | int] = None,
        name: str = 'empgd'
    ) -> None:
        super().__init__(n, epsilon, sigma, rand_start, zor, eta_mask, name)

    def atk(self, model: Vfl, x: Tensor, labels: Tensor) -> None:
        msg = model._x2msg(x)
        _min = torch.min(msg)
        _max = torch.max(msg)
        _epsilon = self.epsilon * (_max - _min)
        _sigma = self.sigma * (_max - _min)

        if self.rand_start:
            eta = torch.randn_like(msg).uniform_(
                -_epsilon.item(), _epsilon.item()
            )
            eta = self.masked(eta)
            adv_msg = msg + eta
            adv_msg.clamp_(_min, _max)
        else:
            eta = torch.zeros_like(msg)
            adv_msg = msg + eta

        def f(_msg: Tensor) -> Tensor:
            y = model.server(_msg)
            loss = self.loss_fn(y, labels)
            return loss

        for _i in range(self.n):
            if self.zor is None:
                adv_msg.requires_grad_()
                adv_msg.retain_grad()
                loss = f(adv_msg)
                msg_grad = torch.autograd.grad(
                    loss, adv_msg,
                    only_inputs=True,
                    retain_graph=False,
                    allow_unused=False
                )[0].detach()
            else:
                msg_grad = self.zor.get_x_grad(adv_msg, f)

            eta += self.masked(msg_grad.sign() * _sigma)
            eta.clamp_(-_epsilon, _epsilon)
            adv_msg = msg + eta
            adv_msg.clamp_(_min, _max)

        with torch.no_grad():
            y = model.server(adv_msg)
            loss = self.loss_fn(y, labels)
        if self.cps is not None:
            self.cps.up(y, labels)
        return y, labels, loss

    def gen_atk_x(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tensor:
        raise ValueError(model)

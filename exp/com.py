import sys
from typing import Iterable, Tuple, Literal
import numpy as np

import torch
from torch import Tensor

TORCH_FLOAT = [torch.float16, torch.float32, torch.float64]


def tensor_size(x: Tensor, bit: int = None) -> int:
    n_element = torch.numel(x)
    if x.dtype in TORCH_FLOAT or bit is None:
        bit = x.element_size() * 8
    return n_element * bit


def tensors_size(y: Iterable[Tensor], bit: int = None) -> int:
    ans = 0
    for x in y:
        ans += tensor_size(x, bit)
    return ans


class Compressor:
    def __init__(
        self,
        typ: Literal[None, 'scale', 'errfb', 'ieee754', 'topk', 'randk'],
        n: int
    ) -> None:
        assert n > 0
        self.typ = typ
        self.n = n

        if typ in ['scale', 'errfb']:
            if n < 32:
                self.dtype = torch.int
            elif n < 64:
                self.dtype = torch.int64
            elif n == 64:
                self.dtype = torch.uint64
            else:
                raise ValueError(n)
            self.num = 2**n
            if typ == 'errfb':
                self.errfb_old = 0.
        elif typ in ['topk', 'randk']:
            self.dtype = None
        elif typ == 'ieee754':
            if n == 16:
                self.dtype = torch.float16
            elif n == 32:
                self.dtype = torch.float32
            elif n == 64:
                self.dtype = torch.float64
            else:
                raise ValueError(n)
        else:
            assert typ is None

    def c_dec(self, x: Tensor) -> Tuple[Tensor, int]:
        x = x.detach()
        if self.typ is None:
            size = tensor_size(x)
            rx = x
        elif self.typ in ['scale', 'errfb']:
            if self.typ == 'errfb':
                x -= self.errfb_old
            low = x.min()
            high = x.max()
            boundaries = torch.linspace(low, high, self.num).to(x.device)
            compressed = torch.bucketize(x, boundaries).to(self.dtype)
            size = tensors_size((compressed, high, low), self.n)

            rx: Tensor = low + compressed * (high - low) / self.num
            if self.typ == 'errfb':
                rx += self.errfb_old
                rx.detach_()
                self.errfb_old = rx
            else:
                rx.detach_()
        elif self.typ in ['topk', 'randk']:
            if self.typ == 'topk':
                _v, _k = x.topk(self.n)

            rx = torch.zeros_like(x)

            if len(x.shape) == 1:
                if self.typ == 'topk':
                    rx[_k] = _v
                else:
                    _k = np.random.choice(x.shape[-1], self.n)
                    rx[_k] = x[_k]
                size = self.n * 32 * 2

            elif len(x.shape) == 2:
                if self.typ == 'topk':
                    for _i in range(x.shape[0]):
                        rx[_i][_k[_i]] = _v[_i]
                else:
                    for _i in range(x.shape[0]):
                        _k = np.random.choice(x.shape[-1], self.n)
                        rx[_i][_k] = x[_i][_k]
                size = self.n * 32 * 2 * rx.shape[0]

            else:
                raise ValueError(len(x.shape))

        elif self.typ == 'ieee754':
            rx = x.to(self.dtype)
            size = tensor_size(rx)
        else:
            raise ValueError(self.typ)
        return rx, size

from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F


class Zor:
    def __init__(
        self,
        q: int,
        mu: float,
        u_coordinate: bool = False,
        u_normalize: bool = True,
        d: float = 10.
    ) -> None:
        self.q = q
        self.mu = mu
        self.u_coordinate = u_coordinate
        self.u_normalize = u_normalize
        if u_normalize:
            self.phi = d
        else:
            self.phi = 1.
        if u_coordinate:
            self.delta_l = q * 2
            self.phi /= 2.
        else:
            self.delta_l = q+1

    def get_x_grad(
        self,
        x: Tensor,
        f: Callable[[Tensor,], Tensor],
        loss: Tensor = None
    ) -> Tensor:
        with torch.no_grad():
            ps = 0.
            if not self.u_coordinate:
                if loss is None:
                    loss = f(x)
            for i in range(self.q):
                u = torch.randn_like(x)
                if self.u_normalize:
                    u = F.normalize(u, dim=-1)
                _l = f(x + self.mu * u)
                if self.u_coordinate:
                    _r = f(x - self.mu * u)
                    delta = _l - _r
                else:
                    delta = _l - loss
                partial = delta.view(-1, 1) * u
                ps += partial
        return self.phi / self.mu * ps / self.q

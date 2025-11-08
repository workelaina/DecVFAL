import random
from itertools import combinations
from typing import Iterable, Tuple

import torch
from torch import Tensor

class Cps_all:
    def __init__(self, n_client: int) -> None:
        self.n_client = n_client
        self.i_clients = list(range(n_client))

    def gt(self) -> Tuple[Iterable[int], int]:
        return self.i_clients, self.n_client

    def up(self, y: Tensor = None, labels: Tensor = None) -> None:
        pass


class Cps_set(Cps_all):
    def __init__(self, i_clients: Iterable[int], n_client: int) -> None:
        super().__init__(n_client)
        self.i_clients = i_clients


class Cps_head(Cps_set):
    def __init__(self, m: int, n_client: int) -> None:
        super().__init__(list(range(m)), n_client)


class Cps_rand(Cps_all):
    def __init__(self, m: int, n_client: int) -> None:
        super().__init__(n_client)
        self.m = m
        self.up()

    def up(self, y: Tensor = None, labels: Tensor = None) -> None:
        self.i_clients = random.sample(range(self.n_client), self.m)


class Cps_ets(Cps_all):
    def __init__(self, m: int, n_client: int) -> None:
        super().__init__(n_client)
        self.m = m
        self.combination = list(combinations(range(n_client), m))
        self.mab = Mab_ets(len(self.combination), 50, torch.device('cuda:0'))
        self.i_clients = self.combination[self.mab.cts_sample()]

    def up(self, y: Tensor, labels: Tensor) -> None:
        _, predicted = torch.max(y, 1)
        self.mab.cts_update(((predicted != labels).sum() / labels.shape[0]).item())
        self.i_clients = self.combination[self.mab.cts_sample()]


class Mab_ets:
    def __init__(self, n_arm: int, warm_round: int, device: str):
        self.n_arm = n_arm
        self.warm_round = warm_round
        self.device = device
        self.round = 0
        self.last_indice = ''
        # n
        self.choice_num = torch.zeros(n_arm, dtype=torch.int, device=device)
        # mu
        self.mean = torch.zeros(n_arm, dtype=torch.float32, device=device)
        # sigma
        self.std = torch.ones(n_arm, dtype=torch.float32, device=device)
        # rmax
        self.upper = torch.zeros(n_arm, dtype=torch.float32, device=device)
        # phi
        self.emp = torch.ones(n_arm, dtype=torch.float32, device=device)

    def cts_sample(self) -> int:
        self.round += 1
        emp_mask = (self.choice_num >= self.round / self.n_arm)
        sample_mask = torch.where(emp_mask == True, 1, 0)
        max_mu, k_max = torch.max(torch.mul(sample_mask, self.mean), 0)
        competitive = self.emp >= max_mu
        competitive[k_max] = True
        competitive = torch.where(competitive == True, 1, 0)

        if self.round > self.warm_round:
            sample = torch.normal(self.mean, self.std)
            sample = torch.mul(sample, competitive)
            indice = torch.max(sample, 0)[1]
        else:
            sample = torch.normal(self.mean, self.std)
            indice = torch.max(sample, 0)[1]

        assert self.last_indice == ''
        self.last_indice = indice

        return indice

    def cts_update(self, grad: float) -> None:
        indice = self.last_indice
        self.last_indice = ''
        # n = n + 1
        self.choice_num[indice] += 1
        # mu
        self.mean[indice] += (grad - self.mean[indice]) / self.choice_num[indice]
        # sigma = 1 / (n+1)
        self.std[indice] = 1 / (self.choice_num[indice] + 1)
        # rmax = max(rmax, r)
        self.upper[indice] = self.upper[indice] if self.upper[indice] >= grad else grad
        # phi
        self.emp[indice] += (self.upper[indice] - self.emp[indice]) / self.choice_num[indice]

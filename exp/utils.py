import os
import sys
import csv
import time
import random
import numpy as np
import hashlib
from typing import Iterable, Tuple, Literal

import torch
from torch import Tensor

BIT_TO_MB = 2 ** 23
# STR_LOSS = 'loss-'
# STR_ACC = 'acc-'


def setup_seed(lucky_seed: str):
    lucky_seed = int(hashlib.shake_256(
        lucky_seed.encode('utf8')
    ).hexdigest(4), 16)
    os.environ['PYTHONHASHSEED'] = str(lucky_seed)
    random.seed(lucky_seed)
    np.random.seed(lucky_seed)
    torch.manual_seed(lucky_seed)
    torch.cuda.manual_seed(lucky_seed)
    torch.cuda.manual_seed_all(lucky_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_last(
    root: str = '../result',
    fmt: str = 'res_%d.d'
) -> str:
    _l = None
    for i in range(1, 99999):
        root_num = os.path.join(root, fmt % i)
        if not os.path.exists(root_num):
            if _l is None:
                raise ValueError('no exist')
            return _l
        _l = root_num
    raise ValueError('too much')


def find_next(
    root: str = '../result',
    fmt: str = 'res_%d.d'
) -> str:
    for i in range(1, 99999):
        root_num = os.path.join(root, fmt % i)
        if not os.path.exists(root_num):
            os.makedirs(root_num, exist_ok=False)
            return root_num
    raise ValueError('too much')


def split_l(shape_1: int, n_client: int) -> Iterable[int]:
    a = shape_1 // n_client
    l1 = shape_1 - a*n_client
    # l0 = n_client - l1
    # l = [a+1]*l1 + [a]*l0

    b = 0
    ans = [0]
    for _ in range(l1):
        b += a+1
        ans.append(b)
    for _ in range(l1, n_client):
        b += a
        ans.append(b)
    return ans


def split_x(
    x: Tensor,
    n_client: int,
    padding: bool
) -> Iterable[Tensor]:
    if n_client == 1:
        return [x]
    _shape = x.shape
    _split_l = split_l(_shape[-1], n_client)
    _len_shape = len(_shape)
    ans_l = list()
    for i in range(n_client):
        _l = _split_l[i]
        _r = _split_l[i+1]
        if _len_shape == 2:
            if padding:
                ans = torch.zeros_like(x)
                ans[:, _l:_r] = x[:, _l:_r]
            else:
                ans = x[:, _l:_r]
        elif _len_shape == 4:
            assert padding
            ans = torch.zeros_like(x)
            ans[:, :, :, _l:_r] = x[:, :, :, _l:_r]
        else:
            raise ValueError(_shape)
        ans_l.append(ans)
    return ans_l


def mask_x(
    x: Tensor,
    i_client: list,
    n_client: int,
    ori: Tensor = None
) -> Tensor:
    if ori is None:
        ans = torch.zeros_like(x)
    else:
        ans = ori.detach()
    if isinstance(i_client, int):
        i_client = [i_client]
    if n_client == 1:
        return x if 0 in i_client else ans
    _shape = x.shape
    _split_l = split_l(_shape[-1], n_client)
    _len_shape = len(_shape)
    for i in i_client:
        _l = _split_l[i]
        _r = _split_l[i+1]
        if _len_shape == 2:
            ans[:, _l:_r] = x[:, _l:_r]
        elif _len_shape == 4:
            ans[:, :, :, _l:_r] = x[:, :, :, _l:_r]
        else:
            raise ValueError(_shape)
    return ans


class Result:
    def __init__(
        self,
        name: str,
        n_batch_prt: int = 25,
        csv_file: str = None
    ) -> None:
        self.name = name
        self.n_batch_prt = n_batch_prt
        self.i_epoch = -2
        self.re_init()
        self.csv = csv_file
        if csv_file is not None:
            self.append('name', 'epoch', 'loss', 'acc', 'mb', 'time')

    def re_init(self) -> None:
        # print('utils.EpochResult time_start')
        self.ce_size = 0
        self.r_ce_size = 0
        self.ce_num = 0
        self.r_ce_num = 0
        self.correct = 0
        self.running_loss = 0.0
        self.i_batch = 0
        self.len_data = 0
        self.i_epoch += 1
        self.time_start = time.time()

    def batch(self, y: Tensor, labels: Tensor, loss: Tensor) -> None:
        _, predicted = torch.max(y, 1)
        _correct = (predicted == labels).sum().item()
        self.correct += _correct
        _loss = loss.item()
        self.running_loss += _loss
        self.i_batch += 1
        self.len_data += labels.shape[0]
        if self.i_batch % self.n_batch_prt == 0:
            print('%s: Batch: %d, Loss: %.7f, Acc: %.3f%%' % (
                self.name,
                self.i_batch,
                _loss/labels.shape[0],
                _correct*100/labels.shape[0]
            ))

    def end_epoch(self) -> Tuple[str, int, float, float, float, float]:
        # print('[ce-1-epoch]', [
        #     self.ce_num, self.ce_size,
        #     self.r_ce_num, self.r_ce_size,
        #     self.i_batch, self.len_data
        # ])
        # sys.exit(0)
        _mb = (self.ce_size + self.r_ce_size) / BIT_TO_MB
        if self.len_data > 0:
            _loss = self.running_loss / self.len_data
            _acc = self.correct * 100 / self.len_data
        else:
            _loss = 0
            _acc = 0
        _t = time.time() - self.time_start
        self.re_init()
        ans = (self.name, self.i_epoch, _loss, _acc, _mb, _t)
        if self.csv is not None:
            self.append(*ans)
        return ans

    def communicate(self, x: int) -> int:
        self.ce_size += x
        self.ce_num += 1
        return self.ce_size

    def r_communicate(self, x: int) -> int:
        self.r_ce_size += x
        self.r_ce_num += 1
        return self.r_ce_size

    def append(self, *args) -> None:
        with open(self.csv, 'a+') as f:
            csv.writer(f).writerow(args)
            f.flush()
            f.close()


class AdvCfg:
    def __init__(
        self,
        name: Literal[
            'clean', 'pgd', 'ours',
            'freeat', 'freelb',
            'dp',
            'cer', 'fgsm'
        ],
        m: int = None,
        n: int = None,
        epsilon: float = None,
        sigma: float = None,
        rand_start: bool = True
    ) -> None:
        self.name = name
        self.m = m
        self.n = n
        self.epsilon = epsilon
        self.sigma = sigma
        self.rand_start = rand_start

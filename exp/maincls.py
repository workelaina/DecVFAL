import os
import sys
import shutil
from typing import Iterable

import torch

from zo import Zor
from com import Compressor
from dataset import Datasets
from attack import *
from model import Vfl
from utils import Result, AdvCfg, find_next, find_last
from party import Linker, mp
from party import ALGO_BATCH, ALGO_FWD, ALGO_MSG


class My_tester:
    def __init__(
        self,
        res_dir: str,
        dataset: Datasets,
        atks: Iterable[Atk_clean],
        device: str,
        n_batch_prt: int = 25
    ) -> None:
        self.dataset = dataset
        self.atks = [(atk, Result(
            'test_%s' % atk.name,
            n_batch_prt,
            os.path.join(res_dir, 'test_%s.csv' % atk.name)
        )) for atk in atks]
        self.device = device

    def test(self, pth: str) -> None:
        model = Vfl.load_vfl(pth).to(self.device)
        model.eval()
        for data in self.dataset.testloader:
            x, labels = data
            if x.shape[0] != self.dataset.batch_size[1]:
                print('Ignore with size', x.shape[0])
                break
            x = x.to(self.device)
            labels = labels.to(self.device)
            for atk, res in self.atks:
                res.batch(*atk.atk(model, x, labels))
        for atk, res in self.atks:
            _name, _i, _loss, _acc, _mb, _t = res.end_epoch()
            print('%s: Loss: %.7lf, Acc: %.5lf%%' % (_name, _loss, _acc))

    def test_mp(self, pth: str) -> None:
        _mp_p = mp.Process(target=self.test, args=(pth,), daemon=True)
        _mp_p.start()
        return _mp_p

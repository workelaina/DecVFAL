import time
from typing import Iterable

import torch
from torch import Tensor

from .constant import *
from .algorithm import Lw_algo
from .clinet import Lw_layer, Lw_layer_inner
from .server import Lw_czo_top

from utils import Result, AdvCfg
from com import Compressor
from zo import Zor

from model.vfl import Vfl


class Linker:
    def __init__(
        self,
        n_client: int,
        model_name: str,
        model_args: list,
        lrs: list,
        client_split: list,
        lw_gpus: Iterable[torch.device],
        lw_level: int,
        async_flg: bool,
        res: Result,
        compressor: Compressor,
        zor: Zor,
        adv_cfg: AdvCfg
    ) -> None:
        assert 1 <= n_client
        assert model_name in ['mlp', 'resnet']
        assert ALGO_MSG <= lw_level <= ALGO_BATCH

        self.vfl = Vfl(n_client, model_name, model_args)

        client_lw_num = len(lrs) - 1

        self.put_to_algo: Iterable[Queue] = list()
        self.get_from_algo: Iterable[Queue] = list()
        _lw_server = Lw_czo_top(
            'Server',
            lw_gpus[client_lw_num*n_client],
            self.vfl.server,
            lrs[-1],
            res,
            compressor,
            zor,
            n_client,
            'async' if async_flg else 'sync'
        )
        self._l = [_lw_server]
        for i in range(n_client):
            _l = client_split[client_lw_num-1]
            _r = client_split[client_lw_num]
            _lw_client = Lw_layer(
                'Cl_%d_%d' % (client_lw_num, i),
                lw_gpus[client_lw_num*(i+1)-1],
                _lw_server,
                self.vfl.clients[i][_l: _r],
                lrs[client_lw_num-1],
                up_n=i,
                n_down=1,
                mp_flg='sync',
                count_delta=None if lw_level >= ALGO_BATCH else 0
            )
            self._l.append(_lw_client)
            for deepth in range(client_lw_num-1, 1, -1):
                _l = client_split[deepth-1]
                _r = client_split[deepth]
                _lw_client = Lw_layer(
                    'Cl_%d_%d' % (deepth, i),
                    lw_gpus[client_lw_num*i+deepth-1],
                    _lw_client,
                    self.vfl.clients[i][_l: _r],
                    lrs[deepth-1],
                    up_n=0,
                    n_down=1,
                    mp_flg='sync',
                    count_delta=None if lw_level >= ALGO_BATCH else 0
                )
                self._l.append(_lw_client)

            _l = client_split[0]
            _r = client_split[1]
            _lw_client = Lw_layer_inner(
                'Inner_%d' % i,
                lw_gpus[client_lw_num*i],
                _lw_client,
                self.vfl.clients[i][_l: _r],
                lrs[0],
                up_n=0,
                mp_flg=None,
                count_delta=None if lw_level >= ALGO_BATCH else 0
            )
            self._l.append(_lw_client)

            _lw_algo = Lw_algo(
                'Algo_%d' % i,
                lw_gpus[client_lw_num*i],
                _lw_client,
                lw_level,
                (i, n_client),
                adv_cfg,
                count_delta=None if lw_level >= ALGO_BATCH else 0
            )
            self._l.append(_lw_algo)
            self.put_to_algo.append(_lw_algo.mp_q_get_from_down_l[0])
            self.get_from_algo.append(_lw_algo.mp_q_put_to_down_l[0])

    def save(self, pth: str) -> None:
        self.vfl.save_vfl(pth)

    def put(self, x: Tensor, labels: Tensor) -> None:
        if x is None:
            for i in self.put_to_algo:
                i.put((None, labels))
        else:
            for i in self.put_to_algo:
                i.put((x.clone(), labels.clone()))
        for i in self.get_from_algo:
            _t = i.get()
            del _t

    def end_epoch(self) -> None:
        self.put(None, ALGO_EPOCH)

    def end_all(self) -> None:
        self.put(None, ALGO_ALL)
        print('END END END END END END END END')
        time.sleep(1)
        print()

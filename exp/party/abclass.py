import time
import random
from typing import Iterable, Tuple, Literal

import torch
from torch import Tensor

from utils import split_x

from .constant import *


class Lw_0:
    def __init__(
        self,
        name: str,
        device: str,
        n_down: int,
        mp_flg: Literal['async', 'sync', None],
        count_delta: int
    ) -> None:
        assert mp_flg in ['async', 'sync', None]
        self.name = name + ' '*(8-len(name))
        self.device = device
        self.n_down = n_down
        self.mp_flg = mp_flg
        self.count_delta = count_delta
        self.count_time_act = 0.
        self.count_time_wait_up = 0.
        self.prt(0, -1, 'init', device, n_down, mp_flg, count_delta)

        if self.mp_flg:
            if self.mp_flg == 'async':
                self.a_cache = None
                self.a_iter = None
            self.mp_q_put_to_down_l: Iterable[Queue] = list()
            self.mp_q_get_from_down_l: Iterable[Queue] = list()
            for _ in range(self.n_down):
                self.mp_q_put_to_down_l.append(Queue())
                self.mp_q_get_from_down_l.append(Queue())
            self._mp_keep = True
            self._mp_p = mp.Process(target=self._mp_always, daemon=True)
            self._mp_p.start()
            del self._mp_p
        else:
            assert self.n_down == 1
            self.not_mp_cache: Tuple[int, Tensor] = (-65533, None)

    def prt(self, level: int, count: int, *args) -> None:
        if DEBUG >= level:
            print(self.name, count, *args, flush=True)

    def _mp_always(self) -> None:
        '''only use in MP'''
        while self._mp_keep:
            if self.n_down == 1:
                for q in self.mp_q_get_from_down_l:
                    _d = q.get()
            else:
                xs = list()
                _d = None
                act_name = None
                for q in self.mp_q_get_from_down_l:
                    del _d
                    _d = q.get()
                    if act_name is None:
                        act_name = _d[DATA_ACT]
                    else:
                        assert act_name == _d[DATA_ACT]
                    if act_name == ACT_FWD:
                        x: Tensor = _d[DATA_X]
                        xs.append(x.to(self.device))
                if act_name == ACT_FWD:
                    if self.mp_flg == 'async' and self.a_cache is not None:
                        _a_list = list(range(self.n_down))
                        random.shuffle(_a_list)
                        for _i in _a_list:
                            self.a_iter = _i
                            self.a_cache[_i] = xs[_i]
                            del _d[DATA_X]
                            _d[DATA_X] = self.a_cache
                            self.act(_d)
                        del xs
                        del _d
                        continue
                    del _d[DATA_X]
                    _d[DATA_X] = torch.cat(xs, dim=-1)
                    if self.mp_flg == 'async':
                        self.a_cache = _d[DATA_X]
                    del xs
                else:
                    if self.mp_flg == 'async':
                        self.a_cache = None

            self.act(_d)
            del _d

    def act(self, data: dict) -> None:
        _start_act_time = time.time()
        act_name = data[DATA_ACT]
        count = data[DATA_COUNT]
        assert act_name in LIST_ACT

        self.prt(10, count, '{ act start', act_name)

        if act_name == ACT_OURS:
            assert DATA_APPEND_LOG not in data
            assert DATA_LEVEL in data
            assert DATA_X in data
            assert DATA_LABEL not in data
            self.act_ours(data)
        elif act_name == ACT_FWD:
            assert DATA_APPEND_LOG in data
            assert DATA_LEVEL in data
            assert DATA_X in data
            assert DATA_LABEL in data
            self.act_fwd(data)
        elif act_name == ACT_ZG:
            assert DATA_APPEND_LOG not in data
            assert DATA_LEVEL not in data
            assert DATA_X not in data
            assert DATA_LABEL not in data
            self.act_zg(data)
        elif act_name == ACT_STEP:
            assert DATA_APPEND_LOG not in data
            assert DATA_LEVEL not in data
            assert DATA_X not in data
            assert DATA_LABEL not in data
            self.act_step(data)
        elif act_name == ACT_SYNC:
            assert DATA_APPEND_LOG not in data
            assert DATA_LEVEL in data
            assert DATA_X not in data
            assert DATA_LABEL not in data
            self.act_sync(data)

        self.prt(10, count, '} act done', act_name)
        self.count_time_act += time.time() - _start_act_time

    def zg(self) -> None:
        pass

    def step(self) -> None:
        pass

    def sync(self, req_count: int) -> None:
        pass

    def _end_epoch(self) -> None:
        pass

    def _end_all(self) -> None:
        self._mp_keep = False

    def act_fwd(self, data: dict) -> None:
        raise ValueError(data[DATA_ACT])

    def act_ours(self, data: dict) -> None:
        self.act_fwd(data)

    def act_zg(self, data: dict) -> None:
        self.zg()
        self.put_to_up_data(data)

    def act_step(self, data: dict) -> None:
        self.step()
        self.put_to_up_data(data)

    def act_sync(self, data: dict) -> None:
        count = data[DATA_COUNT]
        end_level = data[DATA_LEVEL]

        self.put_to_up_data(data)
        if end_level >= ALGO_EPOCH:
            self._end_epoch()
            self.prt(3, count, 'Act: %.2fs, Wait_grad: %.2fs' % (
                self.count_time_act, self.count_time_wait_up
            ))
        self.sync(count)
        if end_level >= ALGO_ALL:
            self._end_all()
        self.prt(20, count, '0.act_sync.put_to_down_cache(sync)')
        self.put_to_down_cache(count, None)
        self.count_time_act = 0.
        self.count_time_wait_up = 0.

    def put_to_up_data(self, data: dict, y: Tensor = None) -> None:
        pass

    def put_to_down_cache(self, count: int, x_grad: Tensor = None) -> None:
        if self.mp_flg:
            if x_grad is None:
                for q in self.mp_q_put_to_down_l:
                    q.put((count, None))
            else:
                _xs = split_x(x_grad.detach(), self.n_down, False)
                if self.mp_flg == 'async' and self.a_iter is not None:
                    q: Queue = self.mp_q_put_to_down_l[self.a_iter]
                    x: Tensor = _xs[self.a_iter]
                    q.put((count, x.clone()))
                    self.a_iter = None
                else:
                    for x, q in zip(_xs, self.mp_q_put_to_down_l):
                        q.put((count, x.clone()))
        else:
            del self.not_mp_cache
            if x_grad is None:
                self.not_mp_cache = (count, None)
            else:
                self.not_mp_cache = (count, x_grad.detach())


class Lw_1(Lw_0):
    def __init__(
        self,
        name: str,
        device: str,
        up: Lw_0,
        up_n: int,
        n_down: int,
        mp_flg: Literal['async', 'sync', None],
        count_delta: int
    ) -> None:
        self._up_mp_flg = up.mp_flg
        if count_delta is None:
            count_delta = up.count_delta
            if self._up_mp_flg:
                count_delta += 1

        if self._up_mp_flg:
            self._up_mp_q_put_to: Queue = up.mp_q_get_from_down_l[up_n]
            self._up_mp_q_get_from: Queue = up.mp_q_put_to_down_l[up_n]
            self._up_mp_cache = (-65533, None)
            del up
        else:
            self.up = up

        super().__init__(name, device, n_down, mp_flg, count_delta)

    def _put_to_up(self, data: dict) -> None:
        if self._up_mp_flg:
            self._up_mp_q_put_to.put(data)
        else:
            self.up.act(data)

    def _get_from_up_1(self) -> Tuple[int, Tensor]:
        del self._up_mp_cache
        self._up_mp_cache = self._up_mp_q_get_from.get()
        return self._up_mp_cache

    def _get_from_up(
        self,
        req_count: int,
        is_wait: bool
    ) -> Tuple[int, Tensor]:
        _start_get_time = time.time()
        self.prt(20, req_count, '{', 'sync' if is_wait else 'fwd', 'req')
        if self._up_mp_flg:
            _c, _grad = self._up_mp_cache
            _k = [_c]
            while not self._up_mp_q_get_from.empty():
                _c, _grad = self._get_from_up_1()
                _k.append(_c)
            if not is_wait:
                req_count -= self.count_delta
            while _c < req_count:
                _c, _grad = self._get_from_up_1()
                _k.append(_c)

            if is_wait:
                while _grad is not None:
                    _c, _grad = self._get_from_up_1()
                    _k.append(_c)
            else:
                while _grad is None:
                    _c, _grad = self._get_from_up_1()
                    _k.append(_c)
        else:
            _c, _grad = self.up.not_mp_cache
        self.prt(20, req_count, '{', 'sync' if is_wait else 'fwd', 'get', _c)
        self.count_time_wait_up += time.time() - _start_get_time

        return _c, _grad

    def get_from_up_grad(self, req_count: int) -> Tensor:
        _c, _grad = self._get_from_up(req_count, False)
        return _grad.to(self.device)

    def get_from_up_sync(self, req_count: int) -> None:
        self._get_from_up(req_count, True)

    def put_to_up_data(self, data: dict, y: Tensor = None) -> None:
        self.prt(20, data[DATA_COUNT], '{ put_to_up_data', data[DATA_ACT])
        _d = gen_data(data, y)
        self._put_to_up(_d)
        self.prt(20, data[DATA_COUNT], '} put_to_up_data', data[DATA_ACT])

    def act_fwd(self, data: dict) -> None:
        count = data[DATA_COUNT]
        x: Tensor = data[DATA_X]

        self.put_to_up_data(data, x)
        x_grad = self.get_from_up_grad(count)
        assert x_grad is not None
        self.prt(20, count, '1.act_fwd.put_to_down_cache(x_grad)')
        self.put_to_down_cache(count, x_grad)

    def sync(self, req_count: int) -> None:
        self.get_from_up_sync(req_count)

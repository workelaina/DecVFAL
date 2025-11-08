import torch
from torch import Tensor

import torch.multiprocessing as mp
from torch.multiprocessing import Queue

# import multiprocessing as mp
# from multiprocessing import Queue

# DEBUG = 0
DEBUG = 3
# DEBUG = 16
# DEBUG = 32

ACT_ZG = 'a_zg'
ACT_FWD = 'a_f'
ACT_STEP = 'a_st'
ACT_SYNC = 'a_sy'
ACT_OURS = 'a_o'

GRAD_FREE = 1000
GRAD_ONLY_X = 1001
GRAD_BWD = 1002

ALGO_MSG = 2000
ALGO_FWD = 2001
ALGO_BATCH = 2002  # ours
ALGO_EPOCH = 2003
ALGO_ALL = 2004

DATA_COUNT = 'd_c'
DATA_ACT = 'd_a'
DATA_LEVEL = 'd_lv'
DATA_APPEND_LOG = 'd_lg'
DATA_X = 'd_x'
DATA_LABEL = 'd_ls'

LIST_ACT = [ACT_ZG, ACT_FWD, ACT_STEP, ACT_SYNC, ACT_OURS]
LIST_DATA = [
    DATA_COUNT, DATA_ACT, DATA_LEVEL,
    DATA_APPEND_LOG, DATA_X, DATA_LABEL
]


def gen_zg_data(count: int) -> dict:
    assert count > 0
    return {
        DATA_COUNT: count,
        DATA_ACT: ACT_ZG
    }


def gen_step_data(count: int) -> dict:
    assert count > 0
    return {
        DATA_COUNT: count,
        DATA_ACT: ACT_STEP
    }


def gen_sync_data(
    count: int,
    end_level: int
) -> dict:
    assert count > 0
    assert ALGO_MSG <= end_level <= ALGO_ALL
    return {
        DATA_COUNT: count,
        DATA_ACT: ACT_SYNC,
        DATA_LEVEL: end_level
    }


def gen_ours_data(
    count: int,
    grad_level: int,
    x: Tensor
) -> dict:
    assert count > 0
    assert GRAD_FREE <= grad_level <= GRAD_BWD
    x = x.clone()
    return {
        DATA_COUNT: count,
        DATA_ACT: ACT_OURS,
        DATA_LEVEL: grad_level,
        DATA_X: x
    }


def gen_fwd_data(
    count: int,
    append_log: bool,
    grad_level: int,
    x: Tensor,
    labels: Tensor
) -> dict:
    assert count > 0
    assert GRAD_FREE <= grad_level <= GRAD_BWD
    x = x.clone()
    if labels is not None:
        labels = labels.clone()
    return {
        DATA_COUNT: count,
        DATA_ACT: ACT_FWD,
        DATA_APPEND_LOG: append_log,
        DATA_LEVEL: grad_level,
        DATA_X: x,
        DATA_LABEL: labels
    }


def gen_data(data: dict, y: Tensor) -> dict:
    act_name = data[DATA_ACT]
    count = data[DATA_COUNT]
    if act_name == ACT_FWD:
        return gen_fwd_data(
            count,
            data[DATA_APPEND_LOG],
            data[DATA_LEVEL],
            y,
            data[DATA_LABEL]
        )

    if act_name == ACT_OURS:
        return gen_ours_data(count, data[DATA_LEVEL], y)

    assert y is None
    if act_name == ACT_SYNC:
        return gen_sync_data(count, data[DATA_LEVEL])

    if act_name == ACT_ZG:
        return gen_zg_data(count)

    assert act_name == ACT_STEP
    return gen_step_data(count)

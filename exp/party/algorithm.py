import time
from typing import Tuple

import torch
from torch import Tensor

from utils import mask_x, AdvCfg

from .constant import *
from .abclass import Lw_1


class Lw_algo(Lw_1):
    def __init__(
        self,
        name: str,
        device: str,
        up: Lw_1,
        lw_level: int,
        mask: Tuple[int, int],
        adv_cfg: AdvCfg,
        count_delta: int = None
    ) -> None:
        if lw_level <= ALGO_FWD:
            assert count_delta == 0
        self.lw_level = lw_level
        self.mask = mask
        self.adv_cfg = adv_cfg
        self.count = 1
        if self.adv_cfg.name == 'clean':
            self.adv_cfg.n = 0
            self.adv_cfg.rand_start = False
            self.algo = self.algo_pgd
        elif self.adv_cfg.name == 'cer':
            self.adv_cfg.n = 0
            self.adv_cfg.rand_start = True
            assert self.adv_cfg.epsilon is not None
            self.algo = self.algo_pgd
        elif self.adv_cfg.name == 'fgsm':
            self.adv_cfg.n = 1
            self.adv_cfg.rand_start = False
            self.adv_cfg.sigma = self.adv_cfg.epsilon
            assert self.adv_cfg.epsilon is not None
            self.algo = self.algo_pgd
        elif self.adv_cfg.name == 'pgd':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.algo = self.algo_pgd
        elif self.adv_cfg.name == 'dp':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.m is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.dpgd_adv_x = list()
            self.dpgd_time_ext = 0.
            self.dpgd_time_1 = 0.
            self.algo = self.algo_dpgd
        elif self.adv_cfg.name == 'ours':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.m is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.algo = self.algo_ours
        elif self.adv_cfg.name == 'freeat':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.global_eta = None
            self.algo = self.algo_freeat
        elif self.adv_cfg.name == 'freelb':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.adv_cfg.sigma /= self.adv_cfg.n
            self.algo = self.algo_freelb

        #############
        elif self.adv_cfg.name == 'fastat':
            self.adv_cfg.n = 1  # FastAT只使用单步
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None  # 步长通常设置为1.25*epsilon
            self.algo = self.algo_fastat
        elif self.adv_cfg.name == 'atas':
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.algo = self.algo_atas
        elif self.adv_cfg.name == 'ours_adaptive':
            assert self.adv_cfg.n is not None
            assert self.adv_cfg.m is not None
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            self.algo = self.algo_ours_adaptive
        elif self.adv_cfg.name == 'fastatlr':
            # 添加FastAT循环学习率版本的支持
            self.adv_cfg.n = 1  # 同样只使用单步
            assert self.adv_cfg.epsilon is not None
            assert self.adv_cfg.sigma is not None
            # 初始化训练进度跟踪变量
            self.current_epoch = 0
            self.batch_count = 0
            self.total_epochs = getattr(self.adv_cfg, 'total_epochs', 30)
            self.batches_per_epoch = getattr(
                self.adv_cfg, 'batches_per_epoch', 500)
            self.algo = self.algo_fastatlr
        ########

        else:
            raise ValueError(self.adv_cfg.name)
        super().__init__(name, device, up, 0, 1, 'sync', count_delta)

    def masked(self, x: Tensor) -> Tensor:
        _i, _n = self.mask
        return mask_x(x, _i, _n)

    def put_to_up_data(self) -> None:
        raise ValueError('put_to_up_data')

    def act(self, d: Tuple[Tensor, Tensor]) -> None:
        x, labels = d
        if x is None:
            self._sync(labels)
            self.put_to_down_cache(self.count)
        else:
            x = x.to(self.device)
            self.put_to_down_cache(self.count)
            self.algo(x, labels)

        if self.lw_level <= ALGO_BATCH:
            self.count += self.count_delta + 2

    def _sync(self, end_level: int = ALGO_MSG) -> None:
        self._put_to_up(gen_sync_data(self.count, end_level))
        self.sync(self.count)
        if end_level == ALGO_EPOCH and self.adv_cfg.name == 'dp':
            self.prt(0, self.count, 'DP_EXT_TIME', self.dpgd_time_ext)
            self.dpgd_time_ext = 0.

    def zg(self) -> None:
        if self.lw_level == ALGO_MSG:
            self._sync()
        self._put_to_up(gen_zg_data(self.count))

    def step(self) -> None:
        if self.lw_level == ALGO_MSG:
            self._sync()
        self._put_to_up(gen_step_data(self.count))

    def fwd(
        self,
        x: Tensor,
        labels: Tensor,
        grad_lv: int = GRAD_BWD,
        append_log: bool = True
    ) -> None:
        if self.lw_level <= ALGO_FWD:
            self._sync()
        self.count += 1
        self._put_to_up(gen_fwd_data(
            self.count,
            append_log,
            grad_lv,
            self.masked(x).clone(),
            labels
        ))

    def ours(
        self,
        x: Tensor,
        grad_lv: int = GRAD_BWD
    ) -> None:
        self._put_to_up(gen_ours_data(
            self.count,
            grad_lv,
            self.masked(x).clone(),
        ))

    def algo_pgd(self, x: Tensor, labels: Tensor) -> None:
        self.zg()
        if self.adv_cfg.rand_start:
            eta = torch.randn_like(x).uniform_(
                -self.adv_cfg.epsilon, self.adv_cfg.epsilon
            )
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        for _i in range(self.adv_cfg.n):
            self.fwd(adv_x, None if _i else labels, GRAD_ONLY_X, False)

            x_grad = self.get_from_up_grad(self.count)

            eta += x_grad.sign() * self.adv_cfg.sigma
            eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)

        self.zg()
        if self.adv_cfg.n:
            self.fwd(adv_x, None)
        else:
            self.fwd(adv_x, labels)
            _x_grad = self.get_from_up_grad(self.count)
        self.step()

    def algo_dpgd(self, x: Tensor, labels: Tensor) -> None:
        _start = time.time()

        self.zg()
        if self.adv_cfg.rand_start:
            eta = torch.randn_like(x).uniform_(
                -self.adv_cfg.epsilon, self.adv_cfg.epsilon
            )
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        for _i in range(self.adv_cfg.n):
            self.fwd(adv_x, None if _i else labels, GRAD_ONLY_X, False)

            x_grad = self.get_from_up_grad(self.count)

            eta += x_grad.sign() * self.adv_cfg.sigma
            eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)

        self.dpgd_adv_x.append((adv_x, labels))

        if len(self.dpgd_adv_x) == self.adv_cfg.m:
            for adv_x, labels in self.dpgd_adv_x:
                self.zg()
                self.fwd(adv_x, labels)
                _x_grad = self.get_from_up_grad(self.count)
                self.step()
            self.dpgd_adv_x = list()
            self.dpgd_time_ext += self.dpgd_time_1
            self.dpgd_time_1 = 0.
        else:
            self.dpgd_time_1 += time.time() - _start

    def algo_ours(self, x: Tensor, labels: Tensor) -> None:
        self.zg()
        if self.adv_cfg.rand_start:
            eta = torch.randn_like(x).uniform_(
                -self.adv_cfg.epsilon, self.adv_cfg.epsilon
            )
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        for _i in range(self.adv_cfg.m):
            self.fwd(adv_x, None if _i else labels, GRAD_BWD, _i == 0)

            for _j in range(self.adv_cfg.n):
                self.ours(
                    adv_x,
                    GRAD_BWD if _j == self.adv_cfg.n-1 else GRAD_ONLY_X
                )

                x_grad = self.get_from_up_grad(self.count)

                eta += x_grad.sign() * self.adv_cfg.sigma
                eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
                adv_x = x + eta
                adv_x.clamp_(0., 1.)

        self.step()

    def algo_freeat(self, x: Tensor, labels: Tensor) -> None:
        if self.global_eta is None:
            if self.adv_cfg.rand_start:
                eta = torch.randn_like(x).uniform_(
                    -self.adv_cfg.epsilon, self.adv_cfg.epsilon
                )
            else:
                eta = torch.zeros_like(x)
        else:
            eta = self.global_eta

        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        for _i in range(self.adv_cfg.n):
            self.zg()
            self.fwd(adv_x, None if _i else labels, GRAD_BWD, _i == 0)

            x_grad = self.get_from_up_grad(self.count)

            self.step()

            eta += x_grad.sign() * self.adv_cfg.sigma
            eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)

        self.global_eta = eta.detach()

    def algo_freelb(self, x: Tensor, labels: Tensor) -> None:
        self.zg()
        if self.adv_cfg.rand_start:
            eta = torch.randn_like(x).uniform_(
                -self.adv_cfg.epsilon, self.adv_cfg.epsilon
            )
            eta /= torch.sqrt(Tensor([x[0].numel()])).to(eta)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        def project(x: Tensor, eps: Tensor) -> Tensor:
            # project X on the ball of radius eps supposing first dim is batch
            dims = list(range(1, x.dim()))
            norms = torch.sqrt(torch.sum(x*x, dim=dims, keepdim=True))
            return torch.min(norms.new_ones(norms.shape), eps/norms) * x

        for _i in range(self.adv_cfg.n):
            self.fwd(adv_x, None if _i else labels, GRAD_BWD, _i == 0)

            x_grad = self.get_from_up_grad(self.count)

            _norm = torch.norm(x_grad.detach())
            eta += x_grad.detach() * self.adv_cfg.sigma / _norm
            eta = project(eta, self.adv_cfg.epsilon)
            eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
            adv_x = x + eta
            adv_x.clamp_(0., 1.)

        self.step()

    #############
    def algo_fastat(self, x: Tensor, labels: Tensor) -> None:
        self.zg()

        # 随机初始化扰动 - FastAT的关键特点
        eta = torch.randn_like(
            x).uniform_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 前向传播获取梯度
        self.fwd(adv_x, labels, GRAD_ONLY_X, True)

        # 获取梯度
        x_grad = self.get_from_up_grad(self.count)

        # 使用略大于扰动半径的步长 (通常是1.25*epsilon)
        step_size = self.adv_cfg.sigma

        # 执行FGSM步骤
        eta += x_grad.sign() * step_size
        eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 对生成的对抗样本做最终前向传播和反向传播
        self.zg()
        self.fwd(adv_x, labels)
        _x_grad = self.get_from_up_grad(self.count)
        self.step()

    def algo_atas(self, x: Tensor, labels: Tensor) -> None:
        self.zg()

        # 初始化梯度范数记录器（如果尚未初始化）
        if not hasattr(self, 'v'):
            self.v = None
            self.beta = 0.5  # 移动平均的动量参数
            self.c = 0.01    # 防止步长过大的常数

        # 使用前一个epoch的对抗样本作为初始化（类似ATTA）
        # 这里可以修改为从存储的扰动中获取，但为简单起见直接随机初始化
        eta = torch.randn_like(x).uniform_(
            -self.adv_cfg.epsilon, self.adv_cfg.epsilon
        )
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 前向传播获取梯度
        self.fwd(adv_x, labels, GRAD_ONLY_X, True)

        # 获取梯度
        x_grad = self.get_from_up_grad(self.count)

        # 计算梯度范数的移动平均
        grad_norm = torch.norm(x_grad, p=2, dim=tuple(
            range(1, x_grad.dim())), keepdim=True)

        if self.v is None:
            self.v = grad_norm
        else:
            self.v = self.beta * self.v + (1 - self.beta) * grad_norm

        # 计算自适应步长 - 步长与梯度范数成反比
        alpha = self.adv_cfg.sigma / (self.c + torch.sqrt(self.v))

        # 执行FGSM步骤，使用自适应步长
        eta += x_grad.sign() * alpha
        eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 对生成的对抗样本做最终前向传播和反向传播
        self.zg()
        self.fwd(adv_x, labels)
        _x_grad = self.get_from_up_grad(self.count)
        self.step()

    def algo_ours_adaptive(self, x: Tensor, labels: Tensor) -> None:
        self.zg()

        # 初始化梯度范数记录器（如果尚未初始化）
        if not hasattr(self, 'v_ours'):
            self.v_ours = None
            self.beta = 0.5  # 移动平均的动量参数
            self.c = 0.01    # 防止步长过大的常数

        if self.adv_cfg.rand_start:
            eta = torch.randn_like(x).uniform_(
                -self.adv_cfg.epsilon, self.adv_cfg.epsilon
            )
            adv_x = x + eta
            adv_x.clamp_(0., 1.)
        else:
            eta = torch.zeros_like(x)
            adv_x = x + eta

        for _i in range(self.adv_cfg.m):
            self.fwd(adv_x, None if _i else labels, GRAD_BWD, _i == 0)

            for _j in range(self.adv_cfg.n):
                self.ours(
                    adv_x,
                    GRAD_BWD if _j == self.adv_cfg.n-1 else GRAD_ONLY_X
                )

                x_grad = self.get_from_up_grad(self.count)

                # 计算梯度范数的移动平均
                grad_norm = torch.norm(x_grad, p=2, dim=tuple(
                    range(1, x_grad.dim())), keepdim=True)

                if self.v_ours is None:
                    self.v_ours = grad_norm
                else:
                    self.v_ours = self.beta * self.v_ours + \
                        (1 - self.beta) * grad_norm

                # 计算自适应步长 - 步长与梯度范数成反比
                adaptive_sigma = self.adv_cfg.sigma / \
                    (self.c + torch.sqrt(self.v_ours))

                eta += x_grad.sign() * adaptive_sigma
                eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
                adv_x = x + eta
                adv_x.clamp_(0., 1.)

        self.step()

    def algo_fastatlr(self, x: Tensor, labels: Tensor) -> None:
        self.zg()

        # 获取当前迭代次数，用于调整扰动步长
        current_epoch = getattr(self, 'current_epoch', 0)
        total_epochs = getattr(self, 'total_epochs', 30)  # 默认总epoch数

        # 实现类似循环学习率的扰动步长调整
        # 在前半部分epochs线性增加，后半部分线性减少
        if current_epoch < total_epochs / 2:
            # 前半部分线性增加扰动步长
            lr_scale = current_epoch / (total_epochs / 2)
        else:
            # 后半部分线性减少扰动步长
            lr_scale = 1.0 - (current_epoch - total_epochs /
                              2) / (total_epochs / 2)

        # 基础步长和最大步长
        base_step_size = self.adv_cfg.sigma * 0.5
        max_step_size = self.adv_cfg.sigma * 1.25

        # 计算当前步长
        adaptive_step_size = base_step_size + \
            lr_scale * (max_step_size - base_step_size)

        # 随机初始化扰动 - FastAT的关键特点
        eta = torch.randn_like(
            x).uniform_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 前向传播获取梯度
        self.fwd(adv_x, labels, GRAD_ONLY_X, True)

        # 获取梯度
        x_grad = self.get_from_up_grad(self.count)

        # 使用自适应步长
        eta += x_grad.sign() * adaptive_step_size
        eta.clamp_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        adv_x = x + eta
        adv_x.clamp_(0., 1.)

        # 对生成的对抗样本做最终前向传播和反向传播
        self.zg()
        self.fwd(adv_x, labels)
        _x_grad = self.get_from_up_grad(self.count)
        self.step()

        # 更新epoch计数
        if not hasattr(self, 'batch_count'):
            self.batch_count = 0

        self.batch_count = getattr(self, 'batch_count', 0) + 1
        # 假设每个epoch有500个batch（根据实际情况调整）
        batches_per_epoch = getattr(self, 'batches_per_epoch', 500)

        if self.batch_count >= batches_per_epoch:
            self.current_epoch = getattr(self, 'current_epoch', 0) + 1
            self.batch_count = 0
    #################

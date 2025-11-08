# DecVFAL

Accelerated Vertical Federated Adversarial Learning through Decoupling Layer-Wise Dependencies

[[Paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/115884)]
[[Code](https://github.com/workelaina/DecVFAL)]
[[Homepage](https://workelaina.github.io/DecVFAL)]

NeurIPS 2025 (Poster)

## Abstract

Vertical Federated Learning (VFL) enables participants to collaboratively train models on aligned samples while keeping their heterogeneous features private and distributed.
Despite their utility, VFL models remain vulnerable to adversarial attacks during inference.
Adversarial Training (AT), which generates adversarial examples at each training iteration, stands as the most effective defense for improving model robustness.
However, applying AT in VFL settings (VFAL) faces significant computational efficiency challenges, as the distributed training framework necessitates iterative propagations across participants.
To this end, we propose **DecVFAL** framework, which substantially accelerates **VFAL** training through a dual-level **Dec**oupling mechanism applied during adversarial sample generation.
Specifically, we first decouple the bottom modules of clients (directly responsible for adversarial updates) from the remaining networks, enabling efficient *lazy sequential propagations* that reduce communication frequency through delayed gradients.
We further introduce *decoupled parallel backpropagation* to accelerate delayed gradient computation by eliminating idle waiting through parallel processing across modules.
Additionally, we are the first to establish convergence analysis for VFAL, rigorously characterizing how our decoupling mechanism interacts with existing VFL dynamics, and prove that **DecVFAL** achieves an $\mathcal{O}(1/\sqrt{K})$ convergence rate matching that of standard VFLs.
Experimental results show that **DecVFAL** ensures competitive robustness while significantly achieving about $3\sim10\times$ speed up.

## BibTeX

```bibtex
@inproceedings{Accelerated2025TianxingMan,
    author = {Tianxing, Man and Yu, Bai and Ganyu, Wang and Jinjie, Fang and Haoran, Fang and Bin, Gu and Yi, Chang},
    title = {Accelerated Vertical Federated Adversarial Learning through Decoupling Layer-Wise Dependencies},
    year = {2025},
    publisher = {Curran Associates Inc.},
    booktitle = {Proceedings of the 39th International Conference on Neural Information Processing Systems},
    series = {NIPS '25}
}
```

# DecVFAL

Accelerated Vertical Federated Adversarial Learning through Decoupling Layer-Wise Dependencies

[[Homepage](https://workelaina.github.io/DecVFAL)]
[[PDF](https://workelaina.github.io/DecVFAL/static/blob/DecVFAL.pdf)]
[[Code](https://github.com/workelaina/DecVFAL)]

NeurIPS 2025 ([Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/115884))

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

## env and run

### init

```shell
sudo apt update
sudo apt install --upgrade screen tree vim git htop gcc g++ colordiff python-is-python3 byobu -y
# sudo apt upgrade
# sudo apt autoremove

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda update -n base -c defaults conda
```

```shell
git clone https://github.com/workelaina/DecVFAL.git
# scp -r exp ubuntu@ip:~/
```

```shell
vim ~/.bashrc
```

```shell
alias ls="ls --color=auto"
alias la="ls --color=auto -al"
alias l="ls --color=auto -ahlF"
alias diff='colordiff'
alias grep='grep --color=auto'
alias egrep='egrep --colour=auto'
alias fgrep='fgrep --colour=auto'
alias dua="du -sh *"
alias vi='vim'
alias py="python3"

# export HF_HOME=/data/home
# export HF_DATASETS_CACHE=/data/dataset
# export TRANSFORMERS_CACHE=/data/tf
# sudo chmod -R 777 /data

conda deactivate
conda activate decvfal
cd ~/exp
```

### env

```shell
# torch 2.2.1
# cuda 12.1.1
# python 3.11

# conda clean -i
# conda update -n base -c defaults conda

conda create -y -n decvfal python=3.11
conda activate decvfal
which python

# https://pytorch.org/
# conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install pillow matplotlib numpy tqdm pandas scikit-learn scipy -c pytorch -c nvidia -y
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pillow matplotlib numpy==1.26.4 tqdm pandas scikit-learn scipy pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### check

```shell
nvidia-smi
# which python
python -c 'import torch;print(torch.cuda.is_available())'
# lscpu
# lsmem
# lspci
uname -a
```

### run

```shell
screen -R sc1
# cd exp
python train.py
# python eval.py
# Ctrl A D
```

## Citation

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

from maincls import *
import argparse

parser = argparse.ArgumentParser(description="ablation")
parser.add_argument('--nlw', default=3, type=int, required=False, help='ablation: cn_lw')
parser.add_argument('--sp', default=3, type=int, required=False, help='ablation: split')
parser.add_argument('-m', default=6, type=int, required=False, help='ablation: ours outer M')
parser.add_argument('-n', default=8, type=int, required=False, help='ablation: ours inner N')
parser.add_argument('-g', type=int, required=True, help='i_gpu')
args = parser.parse_args()

ABLATION_NLW = args.nlw
ABLATION_SP = args.sp
ABLATION_M = args.m
ABLATION_N = args.n
GPU = args.g

cpu_is_good = True
gpus = [GPU]

dataset_name = 'mnist'

model_name = 'resnet'

n_client = 2
n_epoch = 20
mini_epoch = 999999999
lucky_seed = 'Lucky Seed!!'
lrs = [0.0001] * ABLATION_NLW

adv_cfg = AdvCfg(
    name='ours',
    m=ABLATION_M,
    n=ABLATION_N,
    epsilon=8/255,
    sigma=1/255,
    rand_start=True
)
async_flg = False
lw_level = ALGO_BATCH if adv_cfg.name == 'ours' else ALGO_MSG
# lw_level = ALGO_FWD  # yopo - parallelism
# lw_level = ALGO_FWD  # + pipeline model parallelism
layer_1_only_1 = False

# zor = None
zor = Zor(
    q=100,
    mu=.05,
    u_coordinate=False,
    u_normalize=True,
    d=10.
)

compressor = Compressor(
    typ='scale',
    n=2
)


if __name__ == '__main__':
    # mp.freeze_support()
    mp.set_start_method('spawn')
    print('mp start:', mp.get_start_method())

    assert torch.cuda.is_available()
    gpus = [torch.device('cuda:%d' % i) for i in gpus]
    n_gpus = len(gpus)
    n_lw = len(lrs)
    cn_lw = n_lw - 1
    assert 1 <= n_gpus <= 10
    test_gpu = gpus[0]
    server_gpu = gpus[-1]
    if n_gpus <= 2:
        lw_gpus = [server_gpu] * cn_lw * n_client
    else:
        assert n_gpus >= n_client + 1
        if n_gpus == n_client + 1:
            server_gpu = test_gpu
        lw_gpus = list()
        for i_client in range(n_client):
            lw_gpus += [gpus[i_client+1]] * cn_lw
    lw_gpus.append(server_gpu)
    print(test_gpu, *lw_gpus)

    if dataset_name == 'criteo':
        assert model_name == 'mlp'
        y_size = 2
        model_args = [52, 128, 64, y_size]
        batch_size = (160, 160)  # 4 A4000
        dataset_flat = True
    elif dataset_name == 'mnist':
        y_size = 10
        if model_name == 'mlp':
            model_args = [784, 256, 128, y_size]
            batch_size = (32, 16)  # A4000
            dataset_flat = True
        elif model_name == 'resnet':
            model_args = [1, 18, 3, 1, 1, 128, y_size]
            batch_size = (64, 64)  # test 4090*4
            # batch_size = (32, 16)  # 4090
            dataset_flat = False
    elif dataset_name == 'cifar10':
        y_size = 10
        assert model_name == 'resnet'
        model_args = [3, 18, 3, 1, 1, 128, y_size]
        if n_gpus <= 2:
            batch_size = (16, 80)
        else:
            batch_size = (80, 80)
        dataset_flat = False
    elif dataset_name == 'covertype':
        assert model_name == 'mlp'
        y_size = 7
        model_args = [54, 128, 64, y_size]
        batch_size = (512, 512)  # 4 4090
        dataset_flat = True
    else:
        raise ValueError(dataset_name)
    print('model:', model_name, *model_args)
    print('batch size train test:', *batch_size)

    # split
    assert cn_lw*n_client+1 == len(lw_gpus)
    client_split = [0]
    if model_name == 'mlp':
        assert 1 <= cn_lw <= 4
        len_args = len(model_args) - 2
        assert cn_lw <= len_args

        if cn_lw == 2:
            client_split += [2]
            # layer_one, others
        elif cn_lw == 3:
            client_split += [2, 4]
            # layer_one, 1, 1~
        elif cn_lw == 4:
            client_split += [2, 4, 6]
            # layer_one, 1, 1, 1~
        client_split.append(len_args*2)

    elif model_name == 'resnet':
        assert 1 <= cn_lw <= 6
        if cn_lw == 2:
            client_split += [ABLATION_SP]  # [3, 4, 5, 6, 9]
            # conv1, model.bn1, model.relu,
            # others
        elif cn_lw == 3:
            client_split += [3, 5]
            # conv1, model.bn1, model.relu,
            # model.layer1, model.layer2,
            # model.layer3, model.layer4, avgpool, flatten, model.fc
        elif cn_lw == 4:
            client_split += [3, 5, 6]
            # conv1, model.bn1, model.relu,
            # model.layer1, model.layer2,
            # model.layer3,
            # model.layer4, avgpool, flatten, model.fc
        elif cn_lw == 5:
            client_split += [3, 4, 5, 6]
            # conv1, model.bn1, model.relu,
            # model.layer1,
            # model.layer2,
            # model.layer3,
            # model.layer4, avgpool, flatten, model.fc
        elif cn_lw == 6:
            client_split += [3, 4, 5, 6, 9]
            # conv1, model.bn1, model.relu,
            # model.layer1,
            # model.layer2,
            # model.layer3,
            # model.layer4, avgpool, flatten, model.fc
        client_split.append(10)
    else:
        raise ValueError(model_name)
    if cn_lw >= 2 and layer_1_only_1:
        client_split[1] = 1
    assert len(lrs) == len(client_split)
    print('split:', *client_split)

    if ABLATION_M == 6:
        log_root = find_next()
    else:
        log_root = '../result/m%d_n%d.d' % (adv_cfg.m, adv_cfg.n)
        os.makedirs(log_root, exist_ok=True)
    print(log_root)
    shutil.copy2(
        __file__,
        os.path.join(log_root, os.path.basename(__file__))
    )

    dataset = Datasets(
        name=dataset_name,
        batch_size=batch_size,
        num_workers=os.cpu_count() // 2 if cpu_is_good else 0,
        shuffle=True,
        flat=dataset_flat
    )

    lnk = Linker(
        n_client=n_client,
        model_name=model_name,
        model_args=model_args,
        lrs=lrs,
        client_split=client_split,
        lw_gpus=lw_gpus,
        lw_level=lw_level,
        async_flg=async_flg,
        res=Result(
            name='train',
            n_batch_prt=25,
            csv_file=os.path.join(log_root, 'train.csv')
        ),
        compressor=compressor,
        zor=zor,
        adv_cfg=adv_cfg
    )

    i_batch = 0
    for i_epoch in range(n_epoch):
        pth = os.path.join(log_root, 'epoch_%d.pt' % (i_epoch+1))
        print('epoch', i_epoch+1)
        for data in dataset.trainloader:
            x, labels = data
            if x.shape[0] != batch_size[0]:
                print('Ignore with size', x.shape[0])
                break

            if not i_batch:
                lnk.end_epoch()

            lnk.put(x, labels)

            i_batch += 1
            if i_batch % mini_epoch == 0:
                pth = os.path.join(
                    log_root,
                    'epoch_%d_%d.pt' % (i_epoch, i_batch)
                )
                lnk.end_epoch()
                lnk.save(pth)
        lnk.end_epoch()
        lnk.save(pth)
    lnk.end_all()

from maincls import *

cpu_is_good = False
# gpus = [0]
# gpus = range(2)
# gpus = range(4, 8)
# gpus = range(2, 4)
gpus = range(torch.cuda.device_count())

# dataset_name = 'criteo'
# dataset_name = 'mnist'
dataset_name = 'xray'
# dataset_name = 'cifar10'
# dataset_name = 'cifar100'
# dataset_name = 'fer'
# dataset_name = 'covertype'

# model_name = 'mlp'
model_name = 'resnet'
# model_name = 'vgg'
# model_name = 'alexnet'
# model_name = 'cnn'

n_client = 1
n_epoch = 100
mini_epoch = 999999999
lucky_seed = 'Lucky Seed!!'
lrs = [0.0001, 0.0001, 0.0001]

adv_cfg = AdvCfg(
    name='ours_adaptive',
    m=6,
    n=8,
    epsilon=8/255,
    sigma=1/255,
    rand_start=True
)
async_flg = False
# lw_level = ALGO_BATCH if adv_cfg.name == 'ours' else ALGO_MSG
lw_level = ALGO_BATCH if adv_cfg.name.startswith('ours') else ALGO_MSG
# lw_level = ALGO_FWD  # yopo - parallelism
# lw_level = ALGO_FWD  # + pipeline model parallelism
layer_1_only_1 = False

zor = None
# zor = Zor(
#     q=100,
#     mu=.05,
#     u_coordinate=True,
#     u_normalize=True,
#     d=10.
# )

compressor = Compressor(
    typ=None,
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
    elif n_gpus*2 <= n_client:
        server_gpu = test_gpu
        lw_gpus = list()
        for i_client in range(n_client):
            lw_gpus += [gpus[i_client%(n_gpus-1)+1]] * cn_lw
    else:
        if n_gpus <= n_client + 1:
            server_gpu = test_gpu
        lw_gpus = list()
        for i_client in range(n_client):
            lw_gpus += [gpus[(i_client+1)%n_gpus]] * cn_lw
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
            if n_client > 20:
                model_args = [784, 128, 64, y_size]
            batch_size = (256, 128)  # ?
            dataset_flat = True
        elif model_name == 'resnet':
            model_args = [1, 18, 3, 1, 1, 128, y_size]
            # batch_size = (64, 64)  # ablation 4090
            batch_size = (256, 128)  # test 4090*4
            dataset_flat = False
    elif dataset_name == 'xray':
        y_size = 2
        batch_size = (8, 8)
        if model_name == 'mlp':
            model_args = [1024, 512, 256, y_size]
            # if n_client > 16:
            #     raise ValueError('wtf?')
            dataset_flat = True
        elif model_name == 'resnet':
            model_args = [1, 18, 3, 1, 1, 256, y_size]
            # if n_client > 16:
            #     raise ValueError('wtf?')
            dataset_flat = False
    elif dataset_name == 'cifar10':
        y_size = 10
        assert model_name == 'resnet'
        model_args = [3, 18, 3, 1, 1, 128, y_size]
        # model_args = [3, 101, 7, 2, 3, 256, y_size]
        if n_gpus <= 2:
            batch_size = (16, 80)
        else:
            batch_size = (80, 80)
        dataset_flat = False
    elif dataset_name == 'cifar100':
        y_size = 100
        assert model_name == 'resnet'
        model_args = [3, 50, 3, 1, 1, 1000, y_size]
        if n_gpus <= 2:
            batch_size = (80, 160)
        else:
            batch_size = (320, 160)
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
        assert 1 <= cn_lw <= 5
        if cn_lw == 2:
            client_split += [3]
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
        client_split.append(10)
    else:
        raise ValueError(model_name)
    if cn_lw >= 2 and layer_1_only_1:
        client_split[1] = 1
    assert len(lrs) == len(client_split)
    print('split:', *client_split)

    log_root = find_next()
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

    tester = My_tester(
        res_dir=log_root,
        dataset=dataset,
        atks=[
            Atk_clean(),
            Atk_fgsm(96/255, None),
            Atk_pgd(40, 64/255, 2/255, True, None)
        ],
        device=test_gpu
    )

    i_batch = 0
    p: mp.Process = None
    pth = ''

    def saveval(p: mp.Process):
        if p is not None:
            p.join()
            p.close()
            del p
            p = None
        p = tester.test_mp(pth)
        if n_gpus <= n_client + 1:
            p.join()
            p.close()
            del p
            p = None


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
                saveval(p)
                # lnk.end_all()
                # p.join()
                # p.close()
                # del p
                # sys.exit(0)
        lnk.end_epoch()
        lnk.save(pth)
        # saveval(p)
    lnk.end_all()
    p.join()
    p.close()
    del p

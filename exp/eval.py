from maincls import *

if __name__ == '__main__':
    # mp.freeze_support()
    # mp.set_start_method('spawn')
    # print('mp start:', mp.get_start_method())

    n_client = 7
    m_client = 3
    # pth = find_last(find_last(), 'epoch_%d.pt')
    pth = '/hy-tmp/result/MNIST-yopo-resnet-98.72/epoch_12.pt'
    y_size = 10

    log_root = find_next(fmt='test_%d.d')
    shutil.copy2(
        __file__,
        os.path.join(log_root, os.path.basename(__file__))
    )

    t = My_tester(
        res_dir=log_root,
        dataset=Datasets(
            name='mnist',
            batch_size=(128, 128),
            num_workers=0,
            shuffle=True,
            flat=False
        ),
        atks=[
            Atk_clean(),
            Atk_cer(8/255),
            Atk_fgsm(8/255, None, None, 'fgsm'),
            Atk_fgsm(8/255, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), None, 'fgsm-zoo'),
            Atk_pgd(40, 8/255, 1/255, True, None),
            Atk_pgd(40, 8/255, 1/255, True, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), None, 'pgd-zoo'),
            Atk_aa(40, 100, y_size, 8/255, True),
            Atk_cw(100, 20, y_size, 1/255, 1., 1e10, True, None),
            Atk_cw(100, 20, y_size, 1/255, 1., 1e10, True, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), None, 'cw-zoo'),
            Atk_empgd(
                10, 8/255, 1/255, True, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), Cps_head(m_client, n_client), 'embed-head'),
            Atk_empgd(
                10, 8/255, 1/255, True, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), Cps_rand(m_client, n_client), 'embed-rand'),
            Atk_empgd(
                10, 8/255, 1/255, True, Zor(
                q=10,
                mu=.05,
                u_coordinate=True,
                u_normalize=False,
                d=y_size
            ), Cps_ets(m_client, n_client), 'embed-ets')
        ],
        device=torch.device('cuda:0')
    )

    print('eval:', pth)
    t.test(pth)

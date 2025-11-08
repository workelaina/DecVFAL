from maincls import *

if __name__ == '__main__':
    y_size = 10
    ps = list()
    for _pt in range(20, 0, -1):
        for _m in [8, 3]:
            for _n in range(1, 25+1):
                ps.append('../result/m%d_n%d.d/epoch_%d.pt' % (
                    _m, _n, _pt
                ))
    y_size = 10

    log_root = find_next(fmt='mall_%d.d')
    print(log_root)
    shutil.copy2(
        __file__,
        os.path.join(log_root, os.path.basename(__file__))
    )

    t = My_tester(
        res_dir=log_root,
        dataset=Datasets(
            name='mnist',
            batch_size=(256, 256),
            num_workers=0,
            shuffle=False,
            flat=False
        ),
        atks=[
            Atk_pgd(10, 32/255, 4/255, True, None)
        ],
        device=torch.device('cuda:0'),
        n_batch_prt = 999999
    )

    for pth in ps:
        print('eval:', pth)
        if not os.path.exists(pth):
            print('skip')
            continue
        t.test(pth)

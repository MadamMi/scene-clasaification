train_cfg = dict(
    per_batch_size=128,
    cuda=True,
    data_set_dir='/home/hanyu/card_classify/data_card/',
    #init_lr=2.048e-3,
    init_lr=2.048e-2,
    test_init_lr=5e-7,
    #init_lr=5e-7,
    #init_lr=0.02,
    min_lr=1e-7,
    num_epochs=120,
    print_que=1,
    warmup=7186,
    classify=2,
)

model_cfg = dict(
    imagenet_classes=1000,
    data_classes=138,
    if_pre_train=True,
    pre_trained_path='models/peleenet_acc7208.pth.tar',
    test_model_path='testmodel_best.pth.tar',
    #pre_trained_path='testmodel_best.pth.tar',
)

data_cfg = dict(
    iterations_per_epoch=1838,
)


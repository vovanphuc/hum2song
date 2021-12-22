class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 5000
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    meta_train = '/preprocessed/train_meta.csv'
    train_root = '/preprocessed'
    train_list = 'full_data_train.txt'
    val_list = 'full_data_val.txt'

    checkpoints_path = 'checkpoints'
    save_interval = 1

    train_batch_size = 32  # batch size

    input_shape = (630, 80)

    mp3aug_ratio = 1.0
    npy_aug = True

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 0  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = '/result/submission.csv'

    max_epoch = 100
    lr = 1e-2  # initial learning rate
    lr_step = 10
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-1

def load_cifar10_data_iter(batch_size=None, path=''):
    from mxnet.io import ImageRecordIter

    train_record = '' + 'train-data-iter'
    val_record = '' + 'val-data-iter'

    r_mean = 123.680
    g_mean = 116.779
    b_mean = 103.939
    mean = int(sum((r_mean, g_mean, b_mean)) / 3)
    scale = 1 / 59.4415

    train_data = ImageRecordIter(
        batch_size         = batch_size,
        data_name          = 'data',
        data_shape         = (3, 32, 32),
        fill_value         = mean,
        label_name         = 'softmax_label',
        label_width        = 1,
        mean_r             = r_mean,
        mean_g             = g_mean,
        mean_b             = b_mean,
        pad                = 4,
        path_imgrec        = train_record,
        preprocess_threads = 16,
        rand_crop          = True,
        rand_mirror        = True,
        scale              = scale,
        shuffle            = True,
        verbose            = False,
    )

    val_data = ImageRecordIter(
        batch_size         = batch_size,
        data_name          = 'data',
        data_shape         = (3, 32, 32),
        label_name         = 'softmax_label',
        label_width        = 1,
        mean_r             = r_mean,
        mean_g             = g_mean,
        mean_b             = b_mean,
        num_parts          = 2,
        part_index         = 0,
        path_imgrec        = val_record,
        preprocess_threads = 16,
        scale              = scale,
        verbose            = False,
    )

    return train_data, val_data

import warnings
import numpy as np


class default_config(object):
    def __init__(self):
        self.gpu = 0
        self.eeg_dataset = '/home/lab505/mind/image_FT_Transformer/data/block/eeg_signals_128_sequential_band_all_with_mean_std.pth'
        self.splits_path = '/home/lab505/mind/image_FT_Transformer/data/block/block_splits_by_image.pth'#外面的main ../../
        # self.splits_path = '../../data/block_splits_by_image.pth'
        self.split_num = 0
        # dir
        self.log_note = None
        self.base_result_dir = '../results'
        self.save_dir = None
        self.model_save_dir = None
        self.best_val_model_dir = None
        self.best_test_model_dir = None
        self.all_model_dir = None
        self.latest_checkpoint_dir = None
        self.config_save_path = None
        self.run_save_dir = None
        self.log_save_path = None

        # train
        self.model = 'ResNet50'
        self.lstm_layers = 1
        self.lstm_size = 128
        self.embedding_size = 128
        self.num_classes = 40
        # self.model_kwargs = {'act': 'LeakyReLU', 'norm': 'ibn', 'output_stride': 8, 'use_aspp': True,
        #                      'a_rates': [6, 12, 18]}
        # self.model_path = '/home/wz/pytorchDeepLearning/New_Promise12/results/PS_UNET_EDGE_AWARE_2019-10-20 08:54:22.728943/models/latest_checkpoint/PS_UNET_EDGE_AWARE_epoch:417_prostatedice:0.9014394192681469.pth'
        self.total_epoch = 3000
        self.model_path = None
        self.fold = 0
        self.s_epoch = 0
        self.current_epoch = 0
        self.init_lr = 7e-5 #1e-3
        self.lr_decay_epoch = 20
        self.lr_decay_eps = 0.5
        self.lr_decay_th = 1e-4
        self.lr_decay_patience = 15 #2
        self.min_lr = 1e-6
        self.train_epoch = 50
        self.weight_decay = 5e-4
        self.train_stop_patience = 50
        # self.batch_size = 16
        self.batch_size = 4
        self.val_batch_size = 6
        self.seed = 424242
        self.deterministic = True
        self.num_workers = 4
        self.val_every_epoch_after = 0
        self.val_freq = 5
        self.MA_alpha = 0.93
        self.MA_train_loss_th = 5e-4
        # data aug
        self.len_of_data = 185
        self.filter_low = True
        # self.input_size = np.array([64, 176, 176])
        # self.target_space = np.array([1.5, 0.625, 0.625])
        # self.order_data = 1
        # self.order_label = 0
        # self.bright_scale = (0.85, 1.15)
        # self.contrast_scale = (0.85, 1.15)
        # self.gamma_scale = (0.85, 1.15)
        # self.shake_ratio = 0.2
        # self.random_resample_scale = (0.85, 1.15)
        # self.shift_range = (0.0, 0.0)
        # self.scale_range = (-0.15, 0.15)
        # self.rotate_range = (-15.0, 15.0)
        # self.aspect_range = (-0.0, 0.0)

        # quick stop
        self.best_MA_train_loss = None
        self.epoch_of_best_MA_train_loss = None
        self.train_loss_MA = None
        self.val_score = None
        self.best_val_acc = None
        self.best_test_acc = None
        self.epoch_of_best_val_score = None
        self.glo_best_test_acc = 0.96


def parse(self, kwargs):
    for k, v in kwargs.items():
        print(f'{k}\t{v}')
        if not hasattr(self, k):
            warnings.warn(f'config has no attr:{k}')
        setattr(self, k, v)


default_config.parse = parse

opt = default_config()

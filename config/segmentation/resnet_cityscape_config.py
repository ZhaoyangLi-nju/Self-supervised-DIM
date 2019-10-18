from config.default_config import DefaultConfig


class RESNET_CITYSCAPE_CONFIG:

    def args(self):
        log_dir = DefaultConfig.ROOT_DIR + '/summary/'

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'PSP'     # FCN UNET
        arch = 'resnet50'
        dataset = 'cityscapes'

        task_name = 'with_seg'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'imagenet'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # mean = [item * 255 for item in mean]
        # std = [item * 255 for item in std]

        multiprocessing = False
        use_apex = True
        gpus = [0,1,2,3,4,5,6,7]
        batch_size_train = 8
        batch_size_val = 8
        sync_bn = True

        niter = 5000
        niter_decay = 35000
        niter_total = niter + niter_decay
        print_freq = niter_total / 500
        no_trans = False  # if True, no translation loss

        if no_trans:
            loss = ['CLS']
            target_modal = None
        else:
            loss = ['CLS', 'SEMANTIC']
            target_modal = 'seg'

        filters = 'bottleneck'
        base_size = (1024, 2048)
        load_size = (512, 1024)
        random_scale = (0.75, 2)
        fine_size = (712, 712)
        lr = 1e-2

        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        which_content_net = 'resnet50'
        content_pretrained = 'imagenet'

        multi_targets = ['seg']
        multi_modal = False
        which_score = 'up'
        norm = 'in'

        evaluate = True  # report mean acc after each epoch
        slide_windows = True
        resume = False
        # resume_path = 'PSP_None_20000.pth'
        resume_path = 'train_epoch_200.pth'
        inference = False

        return {

            'TASK_TYPE': task_type,
            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'SYNC_BN': sync_bn,
            'PRETRAINED': pretrained,
            'DATASET': dataset,
            'MEAN': mean,
            'STD': std,

            'LOG_PATH': log_dir,
            'DATA_DIR': '/home/lzy/cityscapes',
            # 'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/cityscapes',

            'BASE_SIZE': base_size,
            'RANDOM_SCALE_SIZE': random_scale,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'FILTERS': filters,

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 19,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'LR': lr,
            'MULTIPROCESSING_DISTRIBUTED': multiprocessing,
            'USE_APEX': use_apex,

            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'INFERENCE': inference,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }


from config.default_config import DefaultConfig

class RESNET_SUNRGBD_CONFIG:

    def args(self):
        log_dir = DefaultConfig.ROOT_DIR + '/summary/'

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'FCN'
        arch = 'resnet18'
        dataset = 'sunrgbd'

        task_name = 'test'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'place'
        content_pretrained = 'place'
        gpus = '0,1,'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size_train = 30
        batch_size_val = 30
        sync_bn = False

        niter = 5000
        niter_decay = 10000
        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
            target_modal = None
            multi_modal = False
        else:
            loss = ['CLS', 'SEMANTIC']
            target_modal = 'depth'
            # target_modal = 'seg'
            multi_modal = False

        base_size = (256, 256)
        load_size = (256, 256)
        fine_size = (224, 224)
        lr = 2e-4
        filters = 'bottleneck'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False

        unlabeld = False  # True for training with unlabeled data
        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        which_content_net = 'resnet18'

        multi_scale = False
        multi_targets = ['depth']
        # multi_targets = ['seg']
        which_score = 'up'
        norm = 'in'

        resume = False
        resume_path = 'FCN/2019_09_17_13_50_34/FCN_AtoB_5000.pth'

        return {

            'TASK_TYPE': task_type,
            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'SYNC_BN': sync_bn,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'PRETRAINED': pretrained,
            'FILTERS': filters,
            'DATASET': dataset,

            'LOG_PATH': log_dir,
            'DATA_DIR': '/home/lzy/dataset/sunrgbd_seg',
            # 'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/sunrgbd_seg',

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 37,
            'UNLABELED': unlabeld,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,

            'LR': lr,
            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }

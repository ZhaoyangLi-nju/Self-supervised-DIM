import warnings

# base configuration, might be considered as the abstract class
class DefaultConfig:
    # GPU / CPU
    GPU_IDS = None  # slipt different gpus with comma
    WORKERS = 8
    TASK = None
    TASK_TYPE = 'recognition'

    # MODEL
    MODEL = 'contrastive'
    ARCH = 'resnet18'
    PRETRAINED = 'imagenet'
    MEAN = None
    STD = None
    CONTENT_PRETRAINED = 'imagenet'
    NO_TRANS = False  # set True when evaluating baseline
    FIX_GRAD = False
    IN_CONC = False  # if True, change input_nc from 3 to specific ones

    # PATH
    DATASET = None
    DATA_DIR_TRAIN = '/data0/lzy/STL10/unlabeled/'
    DATA_DIR_VAL = '/data0/lzy/STL10/train/'
    DATA_DIR_UNLABELED = '/home/dudapeng/workspace/datasets/nyud2/mix/conc_data/10k_conc_bak'
    SAMPLE_MODEL_PATH = None
    CHECKPOINTS_DIR = './checkpoints'
    ROOT_DIR = '/home/lzy/'
    # ROOT_DIR = '/home/dudapeng/workspace/'
    LOG_PATH = ROOT_DIR + '/summary/'
    DATA_DIR = None
    DATA_SET = None

    # DATA
    DATA_TYPE = 'pair'  # pair | single
    WHICH_DIRECTION = None
    NUM_CLASSES = 10
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_VAL = 256
    LOAD_SIZE = 96
    FINE_SIZE = 64
    FLIP = True
    UNLABELED = False
    FIVE_CROP = False
    FAKE_DATA_RATE = 0.3
    MULTI_SCALE = True
    MULTI_SCALE_NUM = 5
    RANDOM_SCALE_SIZE = None

    # OPTIMIZATION
    SYNC_BN = True
    MULTIPROCESSING_DISTRIBUTED = False
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_POLICY = 'plateau'  # lambda|step|plateau

    # TRAINING / TEST
    USE_APEX = True
    PHASE = 'train'
    RESUME = False
    RESUME_PATH = None
    NO_FC = True
    INIT_EPOCH = True  # True for load pretrained parameters, False for resume the last training
    START_EPOCH = 0
    ROUND = 1
    MANUAL_SEED = None
    NITER = 10
    NITER_DECAY = 40
    NITER_TOTAL = 50
    LOSS_TYPES = ['cls']  # SEMANTIC_CONTENT, PIX2PIX, GAN
    EVALUATE = True
    USE_FAKE_DATA = False
    CLASS_WEIGHTS_TRAIN = None
    PRINT_FREQ = 100
    NO_VIS = False
    CAL_LOSS = True
    SAVE_BEST = False
    INFERENCE = False
    SLIDE_WINDOWS = False

    # classfication task
    ALPHA_CLS = 1

    # translation task
    WHICH_CONTENT_NET = 'vgg11_bn'
    CONTENT_LAYERS = ['l0', 'l1', 'l2']
    NITER_START_CONTENT = 1
    NITER_END_CONTENT = 300
    ALPHA_CONTENT = 1
    WHICH_SCORE = None
    MULTI_MODAL = False

    TARGET_MODAL = None

    # GAN task
    NO_LSGAN = True  # False: least square gan loss, True: BCE loss
    NITER_START_GAN = 1
    NITER_END_GAN = 200
    ALPHA_GAN = 1

    # Pix2Pix
    NITER_START_PIX2PIX = 1
    NITER_END_PIX2PIX = 200
    ALPHA_PIX2PIX = 1

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut {0}".format(k))
            setattr(self, k, v)

    def print_args(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, ':', getattr(self, k))

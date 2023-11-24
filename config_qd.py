from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.QUICKDRAW = edict()
__C.DATASETS.QUICKDRAW.CATEGORY_FILE_PATH = '../quickdraw_dataset/'
__C.DATASETS.QUICKDRAW.N_POINTS = 2048  # This will need to be updated based on QuickDraw specifics
__C.DATASETS.QUICKDRAW.DATA_PATH = './quickdraw_data/'

# Dataset
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'QuickDraw'
__C.DATASET.TEST_DATASET = 'QuickDraw'

# Constants
__C.CONST = edict()
__C.CONST.NUM_WORKERS = 4
__C.CONST.N_INPUT_POINTS = 2048  # Number of points per QuickDraw sketch, may require adjustment
__C.CONST.mode = 'easy'  # This mode concept may not apply to QuickDraw as it is for point clouds

# Directories
__C.DIR = edict()
__C.DIR.OUT_PATH = 'SVDFormer_QuickDraw'
__C.CONST.DEVICE = '0,1'
# __C.CONST.WEIGHTS = 'path/to/quickdraw/weights.pth'

# Network
__C.NETWORK = edict()
__C.NETWORK.step1 = 2
__C.NETWORK.step2 = 4
__C.NETWORK.merge_points = 1024  # This will need to be updated based on QuickDraw specifics
__C.NETWORK.local_points = 1024  # This will need to be updated based on QuickDraw specifics
__C.NETWORK.view_distance = 1.5  # This concept may not apply to QuickDraw sketches

# Train
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.N_EPOCHS = 300
__C.TRAIN.SAVE_FREQ = 5
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.LR_MILESTONES = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP = 2
__C.TRAIN.WARMUP_STEPS = 300
__C.TRAIN.GAMMA = .98
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.WEIGHT_DECAY = 0

# Test
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'  # This will need to be updated based on QuickDraw specifics

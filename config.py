DATASET_PATH='../charades'
OUTPUT_NAME='testscores.txt'

# Constants
NUM_ACTIONS=157
EPOCHS=50
TORCH_DEVICE=0
INTERMEDIATE_TEST=0
TEST_FREQ=1
LOG=False
USE_GPU=True
SAVE_MODEL=True

# Experiment options
FEATURE_SIZE=4096
HIDDEN_SIZE=512
LAMBDA=0
PREDICTION_LOSS='MSE'
USE_LSTM=True
TRANSFORMER='SMOOTH'

# Optimizer options
OPTIMIZER='SGD'
LR=0.001
MOMENTUM=0.9
CLIP_GRAD=False
LR_DECAY=3

# Data options
TEST_CROP_MODE='CenterCrop'
TRAIN_MODE='SINGLE'
NUM_FLOW=10
# Don't tamper with these variables, they are modified within models if needed 
USE_RGB=True
USE_FLOW=True


def print_config():
    print("# Constants")
    print("NUM_ACTIONS=%d"%(NUM_ACTIONS))
    print("EPOCHS=%d"%(EPOCHS))
    print("TORCH_DEVICE=%d"%(TORCH_DEVICE))
    print("INTERMEDIATE_TEST=%d"%(INTERMEDIATE_TEST))
    print("TEST_FREQ=%d"%(TEST_FREQ))
    print("LOG=%s"%(LOG))
    print("USE_GPU=%s"%(USE_GPU))
    print("SAVE_MODEL=%s"%(SAVE_MODEL))

    print("# Experiment options")
    print("FEATURE_SIZE=%d"%(FEATURE_SIZE))
    print("LAMBDA=%f"%(LAMBDA))
    print("PREDICTION_LOSS=%s"%(PREDICTION_LOSS))
    print("USE_LSTM=%s"%(USE_LSTM))
    print("TRANSFORMER=%s"%(TRANSFORMER))

    print("# Optimizer options")
    print("OPTIMIZER=%s"%(OPTIMIZER))
    print("LR=%f"%(LR))
    print("MOMENTUM=%f"%(MOMENTUM))
    print("CLIP_GRAD=%s"%(CLIP_GRAD))
    print("LR_DECAY=%d"%(LR_DECAY))

    print("# Data options")
    print("TEST_CROP_MODE=%s"%(TEST_CROP_MODE))
    print("TRAIN_MODE=%s"%(TRAIN_MODE))

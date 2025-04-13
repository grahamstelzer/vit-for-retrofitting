# files that need config: <model>.py, benchmark.py, train.py
#  possibly main.py if we want to make cli args 
import torch

# batch size
BATCH_SIZE = 32

# img dims
#   - oxford iiit pet: 144x144 (allegedly)
#   - 
HEIGHT = 144
WIDTH = 144

# input channels 
N_CHANNELS = 3

# patch szie
PATCH_SIZE = 4

# embed dim
EMBED_DIM = 32

# number of layers
N_LAYERS = 6

# number of attn heads
N_HEADS = 2

# feedforward dim size
D_FF = 128
D_HIDDEN = 256

# number of output classes
#   - oxford iiit pet: 37 (allegedly)
N_CLASSES = 37

DROPOUT = 0.1
# DATASET = 
# DATA_PATH = 
# LEARNING_RATE = 
# TRAIN_STEPS = 

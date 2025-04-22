# files that need config: <model>.py, benchmark.py, train.py
#  possibly main.py if we want to make cli args 
import torch

# device
DEVICE = "cuda"

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
#   16 trains really fast on 200 epochs
PATCH_SIZE = 16

# embed dim
#   TODO: check this is working correctly
EMBED_DIM = 128

# number of layers
N_LAYERS = 6

# number of attn heads
N_HEADS = 2

# feedforward dim size
D_FF = 128
D_HIDDEN = 256

# number of output classes
#   - oxford iiit pet: 37 (allegedly)
# NOTE: written as out_dim in ViT() init
N_CLASSES = 37

DROPOUT = 0.1
# DATASET = 
# DATA_PATH = 
LEARNING_RATE = 1e-4
# TRAIN_STEPS = 
WEIGHT_DECAY = 0.01

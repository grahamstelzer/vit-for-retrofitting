import sys
import os

# get config from configs folder
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
sys.path.insert(0, config_path)

from vit_base_config import *



# flop calcs:
def calculate_vit_flops(
    batch_size,
    h,
    w,
    n_channels,
    patch_size,
    embed_dim,
    n_layers,
    n_heads,
    d_ff,
    n_classes):


    '''
    patch embeddings:
    '''

    # total number of patches
    n_patches = (h / patch_size) * (w / patch_size)
    flops_per_patch_embed = batch_size * n_patches * (patch_size * patch_size * n_channels) * embed_dim

    print(flops_per_patch_embed)



    '''
    transformer layers:
        - multiheaded attentoin block
        - ff layer
    '''

    # mha block:




calculate_vit_flops(BATCH_SIZE,
                    HEIGHT,
                    WIDTH,
                    N_CHANNELS,
                    PATCH_SIZE,
                    EMBED_DIM,
                    N_LAYERS,
                    N_HEADS,
                    D_FF,
                    N_CLASSES)
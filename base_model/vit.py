# NOTE: base impl. from https://colab.research.google.com/drive/1P9TPRWsDdqJC6IvOxjG2_3QlgCt59P0w?usp=sharing#scrollTo=Ap3RzZa0yZXt
#       found via yt video: 
# TODO: take this base model, tweak to add things like flash attention, lower prec. matmults


import sys
import os

# get config from configs folder
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
sys.path.insert(0, config_path)

from vit_base_config import *


# get images
import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

# make all images to be the same size, then change all to be tensors
class Compose(object):
    # NOTE: short summary: input=list of FUNCTIONS, called within an image dataset and thus applies the FUNCTIONS
    #           to all images inside the dataset
    """
    A class to compose multiple image transformations into a single callable object.
    Attributes:
        transforms (list): A list of transformation functions to be applied sequentially.
    Methods:
        __call__(image, target):
            Applies each transformation in the `transforms` list to the input `image`.
    Usage:
        1. Initialize the `Compose` object with a list of transformation functions.
           Example: `compose = Compose([transform1, transform2, transform3])`
        2. Call the `compose` object with an image and a target.
           Example: `transformed_image, target = compose(image, target)`
        3. The transformations are applied sequentially to the image, and the target is returned unchanged.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

# def show_images(images, num_samples=40, cols=8):
#     """ Plots some samples from the dataset """
#     plt.figure(figsize=(15,15))
#     idx = int(len(dataset) / num_samples)
#     print(images)
#     for i, img in enumerate(images):
#         if i % idx == 0:
#             plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
#             plt.imshow(to_pil_image(img[0]))

from torchvision.transforms import Normalize


to_tensor = [Resize((144, 144)), ToTensor(), Normalize(mean=[0.5]*3, std=[0.5]*3)]
# initially above for good syntax practice i assume but moved down here to make the dataset=
#   make more sense


dataset = OxfordIIITPet(root=".", download=True, transforms=Compose(to_tensor))
# dataset = AdjustLabelDataset(raw_dataset)



# patcjes:
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor


# NOTE : TESTED after the class code
#          with emb_size = 128, later actual usage uses emb_dim = 32
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),

            # apply linear transformation that gives embedding vectors
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

# Run a quick test
# sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
# print("Initial shape: ", sample_datapoint.shape)
# embedding = PatchEmbedding()(sample_datapoint)
# print("Patches shape: ", embedding.shape)




# attn model
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output






class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)




class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

# NOTE: config values used here
# ff = FeedForward(dim=D_FF, hidden_dim=D_HIDDEN)
# ff(torch.ones((1, 5, 128))).shape




class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x



from einops import repeat

# NOTE: config values used here
class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=16, emb_dim=128,
                n_layers=6, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])


# model = ViT()
# print(model)
# model(torch.ones((1, 3, 144, 144)))






from torch.utils.data import DataLoader
from torch.utils.data import random_split


train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

# NOTE: config files used here
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

# TODO: SOMETHING IS WRONG THIS PRINTS: "Training dataset min: tensor(-0.9765) max: tensor(1.)"
# print("Training dataset min:", torch.min(train.dataset[0][0]), "max:", torch.max(train.dataset[0][0]))
# exit()

import torch.optim as optim
import numpy as np
import logging
from datetime import datetime

device = "cuda"
model = ViT().to(device)

# weight initialization
model.apply(model._init_weights)

# NOTE: config values used here
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


criterion = nn.CrossEntropyLoss()


train_loss_curve = []
test_loss_curve = []


# double check dimensions and patch sizes:
print("Embedding dim:", model.patch_embedding.projection[1].out_features)
print("Patch size:", model.patch_embedding.patch_size)


# # check label dist
# from collections import Counter
# label_counts = Counter([label for _, label in train])
# print("Label distribution (train):", label_counts)


# single input test

# small_train_subset = torch.utils.data.Subset(train, range(10))
# small_train_loader = DataLoader(small_train_subset, batch_size=1, shuffle=True)

# one_input, one_label = next(iter(small_train_loader))
# one_input, one_label = one_input[0].unsqueeze(0).to(device), one_label[0].unsqueeze(0).to(device)

# for i in range(20):
#     model.train()
#     optimizer.zero_grad()
#     output = model(one_input)
#     loss = criterion(output, one_label)
#     loss.backward()
#     optimizer.step()
#     print(f"[{i}] Loss: {loss.item()}")

# Set up logging
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Logging initialized. Starting training process.")

epoch_count = 0

# training loop:
for epoch in range(epoch_count, epoch_count + 100):

    logging.info(f"================== Epoch {epoch} Start ==================")

    epoch_losses = []  # for plotting
    model.train()

    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        logging.info(f"Predicted classes: {outputs.argmax(-1).tolist()}")
        logging.info(f"Actual classes:    {labels.tolist()}")

        print(f"Predicted classes: {outputs.argmax(-1).tolist()}")
        print(f"Actual classes:    {labels.tolist()}")

        loss = criterion(outputs, labels)

        logging.info(f"Current Loss: {loss}")

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_train_loss = np.mean(epoch_losses)
    train_loss_curve.append(mean_train_loss)

    if epoch % 5 == 0:
        print(f"[Epoch {epoch}] Train loss: {mean_train_loss:.4f}")


    logging.info(f"================== Epoch {epoch} End ====================")


import matplotlib.pyplot as plt

plt.plot(train_loss_curve, label="Train Loss")
plt.plot(test_loss_curve, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# Save the training log to a file
with open("training.log", "r") as log_file:
    log_content = log_file.read()
    # Generate a unique filename based on the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_log_{current_time}.txt"

    with open(log_filename, "w") as saved_log_file:
        saved_log_file.write(log_content)

    print(f"Training log has been saved to '{log_filename}'.")


inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)
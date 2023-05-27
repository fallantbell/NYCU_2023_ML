import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=24, class_emb_size=512):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    # self.class_emb = nn.Linear(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=64,           # the target image resolution
        # in_channels=3 + num_classes, # direct
        # in_channels=3 + class_emb_size, # linear
        in_channels=3, # embedding
        out_channels=3,           # the number of output channels
        class_embed_type = None,
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        #small
        block_out_channels=(32, 64, 64), 
        down_block_types=( 
            "DownBlock2D",   
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",         # a regular ResNet upsampling block
        ),
    )
    # 楊家成
    self.model = UNet2DModel(
        sample_size = 64,
        in_channels = 3,
        out_channels = 3,
        layers_per_block = 2,
        class_embed_type = None,
        #num_class_embeds = 2325, #C (24, 3) + 1
        block_out_channels = (128, 128, 256, 256, 512, 512), 
        down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    '''embedding'''
    self.model.class_embedding = nn.Linear(24 ,class_emb_size)

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, y):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    # class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
    # class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    '''direct'''
    # class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)

    # Net input is now x and class cond concatenated together along dimension 1
    # net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)


    # Feed this to the unet alongside the timestep and return the prediction
    '''direct'''
    output = self.model(x, t,class_labels = y).sample
    
    return output # (bs, 3, 64, 64)
     
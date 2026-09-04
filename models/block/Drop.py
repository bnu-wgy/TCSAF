import torch
from torch import nn
import torch.nn.functional as F

try:
    from dropblock import LinearScheduler, DropBlock2D
    HAS_DROPBLOCK = True
except ImportError:
    HAS_DROPBLOCK = False
    print("Warning: dropblock library not found, using custom implementation")
    
    # 自定义 DropBlock2D 实现
    class DropBlock2D(nn.Module):
        """自定义DropBlock2D实现"""
        def __init__(self, block_size=7, drop_prob=0.1):
            super(DropBlock2D, self).__init__()
            self.block_size = block_size
            self.drop_prob = drop_prob
        
        def forward(self, x):
            if not self.training or self.drop_prob == 0.:
                return x
            
            gamma = self.drop_prob / (self.block_size ** 2)
            
            batch_size, channels, height, width = x.shape
            
            mask = torch.bernoulli(
                torch.ones((batch_size, channels, 
                           height - self.block_size + 1,
                           width - self.block_size + 1), 
                          device=x.device) * gamma)
            
            mask = F.max_pool2d(
                mask, 
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=self.block_size // 2)
            
            mask = mask[:, :, :height, :width]
            
            mask = 1 - mask
            
            normalize_factor = mask.numel() / (mask.sum() + 1e-7)
            
            return x * mask * normalize_factor
    
    class LinearScheduler:
        def __init__(self, dropblock, start_value=0., stop_value=0.1, nr_steps=5000):
            self.dropblock = dropblock
            self.start_value = start_value
            self.stop_value = stop_value
            self.nr_steps = nr_steps
            self.current_step = 0
            self.dropblock.drop_prob = start_value
        
        def step(self):
            self.current_step += 1
            if self.current_step < self.nr_steps:
                self.dropblock.drop_prob = self.start_value + \
                    (self.stop_value - self.start_value) * self.current_step / self.nr_steps
            else:
                self.dropblock.drop_prob = self.stop_value
        
        def __call__(self, x):
            return self.dropblock(x)


class DropBlock(nn.Module):

    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )

    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()


def dropblock_step(model):

    actual_model = model.module if hasattr(model, "module") else model
    
    decoder = None
    if hasattr(actual_model, "Binary_Decoder"):
        decoder = actual_model.Binary_Decoder
    elif hasattr(actual_model, "Semantic_Decoder"):
        decoder = actual_model.Semantic_Decoder
    else:
        return
    if hasattr(decoder, "drop"):
        decoder.drop.step()

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.block.Base import ChannelChecker
from models.Encoder.CTF import *
from models.block.Base import Conv1Relu
from models.Encoder.mobilesam import build_sam_vit_t
from models.Encoder.moat import *
from models.Decoder.AFPN_CARAFE import AFPN
from models.block.SS22D import SS22DBlock

class TCSAF(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1])) 
        num_classes = opt.num_classes

        self.SAM_Encoder = build_sam_vit_t()
        self.CNN_Encoder = moat_4(use_window=True, num_classes=10)
        self.Semantic_Decoder = AFPN(self.inplanes)
        self.seg_head = nn.Sequential(
            Conv1Relu(self.inplanes, self.inplanes),
            nn.Conv2d(self.inplanes, num_classes, kernel_size=1)
        )
        self.post_ss2d = nn.Sequential(
                    SS22DBlock(self.inplanes)
                )
        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   
        self.check_channels = ChannelChecker(self.SAM_Encoder, self.inplanes, opt.input_size)
        self.fusion4 = CTF(self.inplanes*8, self.inplanes*8, 16, 16, num_classes=num_classes, use_topo_bias=True)
        self.conv4 = Conv1Relu(self.inplanes*16, self.inplanes*8)


    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, _,fa4 = self.SAM_Encoder(xa)  
        fa1, fa2, fa3,fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, _,fb4 = self.SAM_Encoder(xb)
        fb1, fb2, fb3,fb4 = self.check_channels(fb1, fb2, fb3, fb4)
        fa12, fa22, fa32, fa42 = self.CNN_Encoder(xa)
        fb12, fb22, fb32, fb42 = self.CNN_Encoder(xb)

        inentity_a4 = fa4
        inentity_b4 = fb4
        fa4 = self.conv4(torch.cat([inentity_a4, self.fusion4(fa4, fa42)], 1))
        fb4 = self.conv4(torch.cat([inentity_b4, self.fusion4(fb4, fb42)], 1))

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4
        
        seg_feat = self.Semantic_Decoder(ms_feats)
        seg_feat = self.post_ss2d(seg_feat)
        seg_logits = self.seg_head(seg_feat)
        seg_logits = F.interpolate(seg_logits, size=(h_input, w_input), mode='bilinear', align_corners=True)

        return seg_logits


    def _init_weight(self, pretrain=''):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))


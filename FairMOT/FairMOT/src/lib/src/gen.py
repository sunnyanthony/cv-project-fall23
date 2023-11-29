from torch import nn
from FairMOT.FairMOT.src.lib.models.networks.dlav0 import get_pose_net

class gen(nn.Module):
    def __init__(self, num_classes=1, layer_num=34,reid_dim=128, image_size=(640,640)):
        super().__init__()
        self.img_size = image_size
        self.layer_num = layer_num
        self.ltrb = True
        self.reg_offset = True
        self.conf_thres = 0.3
        self.Kt = 500
        self.heads = {'hm': num_classes, 'wh': 2 if not self.ltrb else 4, 'id': reid_dim, 'reg': 2}
        self.head_conv = 256
        self.down_ratio = 4
        self.model = get_pose_net(layer_num, self.heads, self.head_conv, self.down_ratio)
    
    def forward(self, x):
        return self.model(x)
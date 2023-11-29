from torch import nn
import torch
from torch.nn import functional as F
import math

from models.utils import _tranpose_and_gather_feat

__all__ = [
    'Discriminator',
    'Discriminator0',
    'Discriminator1',
]

def block(input_dim: int, output_dim: int, negative_slope=0.2):
    return nn.Sequential(*[
    nn.Linear(input_dim, output_dim),
    nn.LeakyReLU(negative_slope=negative_slope),
    ])

def cblock(input_dim: int, output_dim: int, negative_slope=0.2, convpara=(4, 2, 1)):
    return nn.Sequential(*[
            nn.Conv2d(input_dim, output_dim, *convpara, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    ])

class ConvReshapeLayer(nn.Module):
    def __init__(self, in_channel, input, output):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel*4, kernel_size=4, stride=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channel*4, out_channels=in_channel*8, kernel_size=4, stride=3, padding=1)
        self.linear = nn.Linear(input, output)  # Adjusted size after convolutions

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.linear(x)
        return x

class input_layer(nn.Module):
    def __init__(self, nID, emb_scale=None, emb_dim=128) -> None:
        """
        nID: from JDE dataloader
        """
        super().__init__()
        self.emb_scale = emb_scale if emb_scale is not None else math.sqrt(2) * math.log(nID - 1)
        # this id_mapper should be as same as the classifier in FairMOT's loss function
        # we should use this id_mapper to be the classifier in the FairMOT's loss function in testing phase
        self.nID =nID
        self.id_mapper = nn.Linear(emb_dim, nID)
        self.id_mapper2 = nn.Linear(500, 500*nID)
        self.wh_mapper = ConvReshapeLayer(4 ,32 * 8 * 15, 500 * 4)
        self.reg_mapper = ConvReshapeLayer(2, 16 * 8 * 15, 500 * 2)

    def forward(self, b, wh, hm, reg, reg_mask, ind, ids, groundtruth):
        """
        wh (4, 152, 272), or (500, 4)
        hm (1, 152, 272)
        reg (2, 152, 272), or (500, 2)
        ids (128, 152, 272), or (500)
        reg_mask_gt (500)
        reg_gt (500, 2)
        ind_gt (500)
        wh_gt (500, 4)
        hm_gt (1, 152, 272)
        """

        # keep dimension static
        if not groundtruth:
            id_head = _tranpose_and_gather_feat(ids, ind)
            # sould 12, 500, 128
            id_head = self.emb_scale * F.normalize(id_head)
            id = self.id_mapper(id_head).view(b, 500, -1).contiguous()
            wh = self.wh_mapper(wh)
            reg = self.reg_mapper(reg)
        else:
            id = self.id_mapper2(ids.float()).view(b, 500, -1).contiguous()
            #id = ids.unsqueeze(2).expand(ids.shape[0], ids.shape[1], self.nID)
        
        if False:
            print(
                f"""
                {wh.shape=} ,
                {hm.shape=} ,
                {id.shape=}
                """
            )

        catlist = [torch.flatten(wh, start_dim=1),
                   torch.flatten(hm, start_dim=1),
                   torch.flatten(reg_mask, start_dim=1),
                   torch.flatten(reg, start_dim=1),
                   torch.flatten(ind, start_dim=1),
                   torch.flatten(id, start_dim=1),]
        metadata = torch.cat(catlist, dim=1)

        return metadata.view(b, -1)

class Discriminator0(nn.Module):
    def __init__(self, img_dim: int = 1088*608, hidden_dim: int = 128):
        super().__init__()
        self.discriminator = nn.Sequential(*[
              block(img_dim, hidden_dim*4),
              block(hidden_dim*4, hidden_dim*2),
              block(hidden_dim*2, hidden_dim),
              nn.Linear(hidden_dim, 1),
            ])

    def forward(self, image: torch.Tensor):
        return self.discriminator(image)

class Discriminator1(nn.Module):
    updown_sampling = []
    def __init__(self, emb_scale, emb_dim, nID, hidden_dim, image_dim=64, input: int = 1088*608):
        super().__init__()
        self.first_layer = input_layer(emb_scale, emb_dim, nID)
        self.main = nn.Sequential(
            cblock(3, image_dim * 2),
            cblock(image_dim * 2, image_dim * 4),
            cblock(image_dim * 4, image_dim * 8),
        )

        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, image, wh, hm, reg_mask, reg, ind, ids, groundtruth):
        """
        groundtruth: to identify
        image: from groundtruth
        reg_mask: from groundtruth
        ind: from groundtruth
        wh: both
        hm: both
        reg: both
        id: both
        """
        b, w, h = image.size()
        metadata = self.first_layer(b, wh, hm, reg_mask, reg, ind, ids, groundtruth)
        x = self.main(image)
        output = torch.cat([x.view(b, -1), metadata.view(b, -1)], -1)
        return self.out_layer(output)
    


class Discriminator(nn.Module):
    updown_sampling = []
    def __init__(self, emb_scale, emb_dim, nID, hidden_dim,
                 total_dim=1154036,#500 * 4 + 1 * 152 * 272 + 500 + 500 * 2 + 500,
                 gt_dim=500 + 500 + 500 + 500*2 + 500*4 + 1*152*272):
        super().__init__()
        self.first_layer = input_layer(nID, emb_scale, emb_dim, )
        self.gt_layer = nn.Sequential(
            nn.LayerNorm(gt_dim, eps=1e-12),
            nn.Linear(gt_dim, hidden_dim * 3),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.LayerNorm(total_dim, eps=1e-12),
            nn.GELU(),
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, wh, hm, reg, id, ids_gt, reg_mask_gt, reg_gt, ind_gt, wh_gt, hm_gt, groundtruth):
        """
        wh (4, 152, 272), or (500, 4)
        hm (1, 152, 272)
        reg (2, 152, 272), or (500, 2)
        id (128, 152, 272), or (500)
        ids (500)
        reg_mask_gt (500)
        reg_gt (500, 2)
        ind_gt (500)
        wh_gt (500, 4)
        hm_gt (1, 152, 272)
        """
        if True:
            print(groundtruth)
        batch_size = wh.shape[0] # should be 12
        # metadata.shape should be 500 * 4 + 1 * 152 * 272 + 500 + 500 * 2 + 500 + nID
        metadata = self.first_layer(batch_size, wh, hm, reg, reg_mask_gt, ind_gt, id, groundtruth)
        gt_layer_input = torch.cat([
            torch.flatten(ids_gt, start_dim=1),
            torch.flatten(ind_gt, start_dim=1),
            torch.flatten(reg_mask_gt, start_dim=1),
            torch.flatten(reg_gt, start_dim=1),
            torch.flatten(wh_gt, start_dim=1),
            torch.flatten(hm_gt, start_dim=1)], dim=-1)
        gtdata = self.gt_layer(gt_layer_input)
        output = torch.cat([gtdata.view(batch_size, -1), metadata.view(batch_size, -1)], -1)
        return self.out_layer(output)
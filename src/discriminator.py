from torch import nn
import torch
from torch.nn import functional as F
import math

from FairMOT.FairMOT.src.lib.models.utils import _tranpose_and_gather_feat

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

class input_layer(nn.Module):
    def __init__(self, nID, emb_scale=None, emb_dim=128) -> None:
        """
        nID: from JDE dataloader
        """
        super().__init__()
        self.emb_scale = emb_scale if emb_scale is not None else math.sqrt(2) * math.log(nID - 1)
        # this id_mapper should be as same as the classifier in FairMOT's loss function
        # we should use this id_mapper to be the classifier in the FairMOT's loss function in testing phase
        self.id_mapper = nn.Linear(emb_dim, nID)

    def forward(self, b, wh, hm, reg_mask, reg, ind, ids, groundtruth):
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

        # keep dimension static
        mask = reg_mask > 0
        id = ids[~mask] = -1
        if not groundtruth:
            id_head = _tranpose_and_gather_feat(id, ind)
            id_head = id_head[reg_mask > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id = self.id_mapper(id_head).contiguous()
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
    def __init__(self, emb_scale, emb_dim, nID, hidden_dim, gt_dim, totol_dim):
        super().__init__()
        self.first_layer = input_layer(emb_scale, emb_dim, nID)
        self.gt_layer = nn.Sequential(
            nn.LayerNorm(gt_dim, eps=1e-12),
            nn.Linear(gt_dim, hidden_dim * 2),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.LayerNorm(totol_dim, eps=1e-12),
            nn.GELU(),
            nn.Linear(totol_dim, hidden_dim),
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

    def forward(self, wh, hm, reg, ids, reg_mask_gt, reg_gt, ind_gt, wh_gt, hm_gt, groundtruth):
        """
        groundtruth: to identify
        reg_mask: from groundtruth
        ind: from groundtruth
        wh: both
        hm: both
        reg: both
        id: both
        """
        batch_size = wh.shape[0]
        metadata = self.first_layer(batch_size, wh, hm, reg_mask_gt, reg, ind_gt, ids, groundtruth)
        gt_layer_input = torch.cat([torch.flatten(ind_gt, start_dim=1),
            torch.flatten(reg_gt, start_dim=1),
            torch.flatten(wh_gt, start_dim=1),
            torch.flatten(hm_gt, start_dim=1)], dim=-1)
        gtdata = self.gt_layer(gt_layer_input)
        output = torch.cat([gtdata.view(batch_size, -1), metadata.view(batch_size, -1)], -1)
        return self.out_layer(output)
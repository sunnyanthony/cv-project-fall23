from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer
from src.discriminator import Discriminator, Discriminator0
from src.gan import losses


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.gan = opt.gan
        if self.gan:
            #print(f"{self.emb_scale}, {self.emb_dim}, {self.nID}")
            if True:
                self.D = Discriminator0(self.emb_scale, self.emb_dim, self.nID, 64).to(opt.device)
            else:
                self.D = Discriminator(self.emb_scale, self.emb_dim, self.nID, 64).to(opt.device)
            loss = losses[opt.enable_gan]
            self.D_loss = loss()
            self.D_opt = torch.optim.Adam(self.D.parameters(), betas=(0.5, 0.999))
            self.G_loss = torch.nn.MSELoss()#loss()
        
    def forward(self, outputs, batch):
        if self.gan:
            return self.forward_1(outputs, batch)
        else:
            return self.forward_0(outputs, batch)

    def forward_1(self, outputs, batch):
        opt = self.opt
        loss, hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0, 0
        torch.cuda.empty_cache()

        self.D_opt.zero_grad()
        d_input = [ {key: value.detach() for key, value in output.items()} for output in outputs]
        d_fake_loss, d_real_loss = 0, 0
        groundtruth = False
        for s in range(opt.num_stacks):
            # the fake data should far from zero, because d_out is the error of the predict and ground truth
            input = d_input[s]
            d_out =  self.D(input['wh'],
                            input['hm'],
                            input['reg'],
                            input['id'],
                            batch['ids'],
                            batch['reg_mask'],
                            batch['reg'],
                            batch['ind'],
                            batch['wh'],
                            batch['hm'], groundtruth)
            d_fake_loss += d_out#self.D_loss(d_out, torch.ones_like(d_out) * -1)
        del d_input
        torch.cuda.empty_cache()
        # the fake data should close to zero, because d_out is the error of the predict and ground truth
        groundtruth = True
        input = batch
        d_out =  self.D(input['wh'],
                        input['hm'],
                        input['reg'],
                        input['ids'],
                        batch['ids'],
                        batch['reg_mask'],
                        batch['reg'],
                        batch['ind'],
                        batch['wh'],
                        batch['hm'], groundtruth)
        torch.cuda.empty_cache()
        d_real_loss += 1 - d_out#self.D_loss(d_out, torch.ones_like(d_out))
        d_total = 0.5 * d_real_loss + 0.5 * d_fake_loss
        d_total.mean().backward()
        self.D_opt.step()
        torch.cuda.empty_cache()

        g_fake_loss = 0
        groundtruth = False
        for s in range(opt.num_stacks):
            # the fake data should close to zero, because g_out want to cheat the discriminator
            input = outputs[s]
            g_out =  self.D(input['wh'],
                            input['hm'],
                            input['reg'],
                            input['id'],
                            batch['ids'].detach(),
                            batch['reg_mask'].detach(),
                            batch['reg'].detach(),
                            batch['ind'].detach(),
                            batch['wh'].detach(),
                            batch['hm'].detach(), groundtruth)
            g_fake_loss += g_out#self.G_loss(g_out, torch.ones_like(g_out))
        loss_stats = {'loss': 1 - g_fake_loss, 'hm_loss': hm_loss,
                     'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}

        torch.save(self.D, '/home/grads/anthonycheng/cv-project-fall23/FairMOT/FairMOT/model_jit.pt')

        return 1 - g_fake_loss, loss_stats

    def forward_0(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss']#, 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

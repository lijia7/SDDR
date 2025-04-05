#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:21:19 2018

@author: hongweizou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

__dict__ = ['L1Loss', 'L2Loss','LnAlphaLoss', 'berHuLoss', 'SILogLoss', 'JointVNLLoss', 'JointEdgeRankingLoss']

class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, images=None):
        mask = targets.gt(0).float()
        loss = mask * (inputs - targets).abs()
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss/mask.sum()
        return loss

class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, images=None):
        mask = targets.gt(0).float()
        loss = mask * (inputs - targets).pow(2)
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss/mask.sum()
        return loss    



class LnAlphaLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super(LnAlphaLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets, images=None):
        inputs = inputs[targets>0]
        targets = targets[targets>0]
        xabs = (inputs - targets).abs()
        loss =torch.log(xabs + self.alpha)
        loss = loss.mean()
    
        return loss

class SILogLoss(nn.Module):
    def __init__(self, variance_focus=0.85, reduction='mean'):
        super(SILogLoss, self).__init__()
        self.reduction = reduction
        self.variance_focus = variance_focus

    def forward(self, inputs, targets, mask):
        inputs = inputs[mask!=255.]+1e-5
        targets = targets[mask!=255.]+1e-5
        d = torch.log(inputs) - torch.log(targets)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class berHuLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(berHuLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets, images=None):
        xabs = (inputs - targets).abs()
        c = 0.2*torch.max(xabs)
        mask = targets.gt(0).float()
        norm1mask = xabs.le(c).float()
        norm2mask = xabs.gt(c).float()
        loss = (xabs * norm1mask + (xabs*xabs*norm2mask + c*c)/(2*c))*mask
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss/mask.sum()
    
        return loss



######################################################
# EdgeguidedRankingLoss (with regularization term)
#####################################################

"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling), and IGS (Instance-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B

###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):

    # find edges
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(inputs, edges_mask)
    targets_edge = torch.masked_select(targets, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
    anchors = torch.gather(inputs_edge, 0, index_anchors)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(2, 31, (4,sample_num)).cuda()
    pos_or_neg = torch.ones(4,sample_num).cuda()
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.abs(torch.cos(theta_anchors)).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.abs(torch.sin(theta_anchors)).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = torch.gather(inputs, 0, A.long())
    inputs_B = torch.gather(inputs, 0, B.long())
    targets_A = torch.gather(targets, 0, A.long())
    targets_B = torch.gather(targets, 0, B.long())
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())

    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num

class EdgeguidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=3000, sigma=0.03, alpha=1.0, mask_value=1e-1):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        # self.regularization_loss = GradientLoss(scales=4)

    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    def forward(self, inputs, targets, images):

        masks = targets > 0
        masks = masks.float()

        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)

        #=============================
        n,c,h,w = targets.size()
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
            edges_img = edges_img.view(n, -1).double()
            thetas_img = thetas_img.view(n, -1).double()

        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()
            edges_img = edges_img.contiguous().view(1, -1).double()
            thetas_img = thetas_img.contiguous().view(1, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.]).cuda()
        inputs_edge = []
        minlen = []

        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            # Random Sampling
            random_sample_num = sample_num
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, random_sample_num)

            # Combine EGS + RS
            inputs_A = torch.cat((inputs_A, random_inputs_A), 0)
            inputs_B = torch.cat((inputs_B, random_inputs_B), 0)
            targets_A = torch.cat((targets_A, random_targets_A), 0)
            targets_B = torch.cat((targets_B, random_targets_B), 0)
            masks_A = torch.cat((masks_A, random_masks_A), 0)
            masks_B = torch.cat((masks_B, random_masks_B), 0)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A * masks_B

            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask

            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean() 

        return loss.float()/n


class JointEdgeRankingLoss(nn.Module):
    def __init__(self, loss = 'SILogLoss', gamma = 1.0, reduction='mean'):
        super(JointEdgeRankingLoss, self).__init__()

        self.gamma = gamma
        self.edgeranking = EdgeguidedRankingLoss( point_pairs=3000, sigma=0.03, alpha=1.0, mask_value=1e-1)
        if loss == 'SILogLoss':
            self.loss = SILogLoss(variance_focus=0.85, reduction=reduction)
        elif loss == 'LnAlphaLoss':
            self.loss = LnAlphaLoss(alpha=0.5, reduction=reduction)

    def forward(self, inputs, targets, images):
        loss1 = self.loss(inputs, targets) 

        loss2 = self.edgeranking(inputs, targets, images)

        print(loss1.item(), loss2.item())

        loss = loss1 + self.gamma * loss2

        return loss








# VNL Loss
class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #[]
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss


class JointVNLLoss(nn.Module):
    def __init__(self, loss = 'SILogLoss', gamma = 2.0, reduction='mean'):
        super(JointVNLLoss, self).__init__()

        self.gamma = gamma
        self.vnl = VNL_Loss(focal_x=357.3, focal_y=357.3, input_size=(256, 640),
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15)
        if loss == 'SILogLoss':
            self.loss = SILogLoss(variance_focus=0.85, reduction=reduction)
        elif loss == 'LnAlphaLoss':
            self.loss = LnAlphaLoss(alpha=0.5, reduction=reduction)

    def forward(self, inputs, targets, images=None):
        loss1 = self.loss(inputs, targets) 

        loss2 = self.vnl(inputs, targets)

        loss = loss1 + self.gamma * loss2

        return loss






















# classification loss is currently incomplete

###==================================  classication ==================================================###

#########################################################################################################
#focal loss
#########################################################################################################
class FocalLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=2, gamma = 3):
        super(FocalLoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        #self.nll_loss = nn.NLLLoss2d(weight, size_average,reduce=False)

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()
        prob = F.softmax(inputs)

        class_mask = torch.zeros(inputs.size())
        target_temp = targets.cpu()
        target_temp = target_temp.data.numpy()
        for k in np.arange(0,n):
            for i in np.arange(0,h):
                for j in np.arange(0,w):
                    class_mask[k,target_temp[k,0,i,j],i,j]=1
        class_mask = Variable(class_mask).cuda()
        loss_t = -prob.log()*class_mask
        loss_t = self.alpha*(torch.pow((1-prob), self.gamma))*loss_t
        #add mask for depth
        mask = targets.gt(0).float()
        loss_t=  mask *torch.sum(loss_t)/(h*w)

        #nllloss = self.nll_loss(F.log_softmax(inputs), targets)
        #print('NLLL:',nllloss.sum()/(w*h), 'mine:',loss_t)
        del target_temp,class_mask, mask
        return loss_t

class Entropy2dLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Entropy2dLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average,reduce=False)

    def forward(self, inputs, targets):
        #add mask for depth
        mask = targets.gt(0).float()
        loss_t = self.nll_loss(F.log_softmax(inputs), targets)
        loss_t = loss_t * mask 
        loss_t = loss_t.mean()

        return loss_t
    
###==================================  regression ==================================================###








class Balanced_MAELoss(nn.Module):
    def __init__(self):
        super(Balanced_MAELoss, self).__init__()


    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()
        mask = targets.gt(0).float()
        max_data = torch.max((1e-6+inputs).log(), (1e-6+targets).log())
        min_data = torch.min((1e-6+inputs).log(), (1e-6+targets).log())
        lamda_D = 1 - mask*min_data/(max_data + 1e-6)
        alpha_D = torch.exp(targets/10.0)

        res = (inputs - targets).abs()
        mask1 = res.le(1).float()
        loss = ((lamda_D/10 + alpha_D) * (0.5*res.pow(2)*mask1 + (res-0.5)*(1-mask1)) )*mask
        loss = loss.sum() / mask.sum()
        
        return loss


#########################################################################################################
#berHu loss
#########################################################################################################
class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()
        
    def forward(self, inputs, targets):
        xabs = (inputs - targets).abs()
        c = 0.2*torch.max(xabs)
        mask = targets.gt(0).float()
        norm1mask = xabs.le(c).float()
        norm2mask = xabs.gt(c).float()
        
        loss = (xabs * norm1mask + (xabs*xabs*norm2mask + c*c)/(2*c))*mask
        loss = loss.sum()/mask.sum()
    
        return loss



#########################################################################################################
#Turkey biweight loss
#########################################################################################################    
class TurkeyBiweightLoss(nn.Module):
   def __init__(self):
       super(TurkeyBiweightLoss, self).__init__()
       self.c = 4.6851
       
   def forward(self, inputs, targets):
        n,c,h,w = targets.size()
        mask = targets.gt(0).float()

        max_data = torch.max((1e-6+inputs).log(), (1e-6+targets).log())
        min_data = torch.min((1e-6+inputs).log(), (1e-6+targets).log())
        lamda_D = 1 - mask*min_data/(max_data + 1e-6)


        mask1 = (targets.gt(0).float().mean(0)).eq(1).float()

        num = mask1.sum()*n 


        deltas = mask1*(targets - inputs)

        MAD = 1.4826*(torch.median((torch.abs(deltas - (torch.median(deltas, 0)[0]))),0)[0])

        invalidnum = w*h*n - num

        Non_zeros = (torch.sum(((deltas.abs()).lt(4.6851).float()*mask1)))/num

        if Non_zeros < 0.85:
            print('Non_zeros:', Non_zeros.cpu().data[0].numpy())
            MAD = 7.0*MAD  #helps the convergence at the first iterations

        R_mad = deltas   #/(MAD+1e-6)

        loss =torch.clamp(((self.c*self.c)/6.0*(1 - (1 - (R_mad/self.c).pow(2)).pow(3))), max=(self.c*self.c)/6.0) #+ lamda_D*R_mad.abs()

        loss = loss.sum()/num 

        return torch.sqrt(loss)    


#########################################################################################################
#Improved Ranking loss
#########################################################################################################
class ImprovedRankingLoss(nn.Module):
    def __init__(self):
        super(ImprovedRankingLoss, self).__init__()
        self.samplenum = 4000
        self.sigma = 0.03
       
    def forward(self, inputs, targets):
        n,c,h,w = targets.size()
        if n != 1:
            inputs = inputs.view(n,-1).double()
            targets = targets.view(n,-1).double()
        else:
            inputs = inputs.contiguous().view(1,-1).double()
            targets = targets.contiguous().view(1,-1).double()

        inputs_nonzero =[]
        targets_nonzero = []

        minlen = []
        for i in range(0,n):
            inputs_nonzero.append(torch.masked_select(inputs[i],(targets[i]).gt(0)))
            targets_nonzero.append(torch.masked_select(targets[i],(targets[i]).gt(0)))
            minlen.append(inputs_nonzero[i].size()[0])

        loss = torch.DoubleTensor([0.0]).cuda()

        for i in range(0,n):
            indexs=torch.randint(0,minlen[i], (self.samplenum*2,), dtype=torch.long).cuda()
            inputs_sample = torch.gather(inputs_nonzero[i], 0, indexs).view(2,-1)
            targets_sample = torch.gather(targets_nonzero[i], 0, indexs).view(2,-1)
            #print 'inputs sample:', inputs_sample
            #print 'targets_sample', targets_sample
            ratio = targets_sample[0]/(targets_sample[1]+1e-6)
            label = ratio.gt(1+self.sigma).double() - ratio.lt(1/(1+self.sigma)).double()
            #print 'binary label:', label
            mask = label.ne(0).double()
            phi_ne = torch.log(1 + torch.exp((-inputs_sample[0] + inputs_sample[1])*label))*mask 
            #print 'phi_ne', phi_ne
            phi_ne = torch.sort(phi_ne)[0]
            #print 'sort:', torch.sort(phi_ne)
            phi_ne[0:len(phi_ne)/4]=0
            #print 'zeros:', phi_ne

            phi_eq = (inputs_sample[0]- inputs_sample[1]).pow(2)*(1-mask)
            #print 'phi_eq:', phi_eq

            loss += (phi_ne.sum() + phi_eq.sum())/(len(phi_ne)*3/4 + len(phi_eq))

        return loss.float()/n






def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch,colorama
colorama.init(autoreset=True)
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
'''
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input
'''

class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



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
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(1), target.squeeze(1), mask.squeeze(1)
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss_ = 0

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask]) + loss_
        if not return_interpolated:
            return loss
        return loss, intr_input, torch.abs(scaled_prediction-target)

class ScaleAndShiftInvariantLoss_trimed(nn.Module):
    def __init__(self,trim=0.2):
        super().__init__()
        self.name = "SSILoss"
        self.trim = trim
        self.trimmaeloss=TrimmedMAELoss

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(1), target.squeeze(1), mask.squeeze(1)
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        loss = self.trimmaeloss(scaled_prediction,target,mask,self.trim)
        if not return_interpolated:
            return loss
        return loss, intr_input, torch.abs(scaled_prediction-target)

def TrimmedMAELoss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))         # valid pixel numbers
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]

    return trimmed.sum() / (2 * M.sum())

def normalize_prediction_robust(target, mask):
    
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum,dtype=torch.float32)
    s = torch.ones_like(ssum,dtype=torch.float32)

    # print(m.dtype,mask.dtype,target.dtype)
    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)
    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    return target / (s.view(-1, 1, 1))

def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

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

def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

class mygradloss(nn.Module):
    def __init__(self, kernalsize=448):
        super().__init__()
        device = torch.device('cuda')
        self.kernalsize=kernalsize+1
        self.margin=(self.kernalsize-1)//2
        self.pool = nn.MaxPool2d(kernel_size=(self.kernalsize, self.kernalsize),stride=1)
        self.sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        
        self.sobel_x=self.sobel_x.to(device)
        self.sobel_y=self.sobel_y.to(device)
        self.pool=self.pool.to(device)
        self.L=TrimmedProcrustesLoss(alpha=0)

    def forward(self, pred, target_grad, grad_mask):
        pred_grad_x = F.conv2d(pred, self.sobel_x, stride=1, padding=1,)
        pred_grad_y = F.conv2d(pred, self.sobel_y, stride=1, padding=1,)
        pred_grad=torch.abs(pred_grad_x)+torch.abs(pred_grad_y)

        loss_grad = self.L(pred_grad.squeeze().to(torch.float32), target_grad.squeeze().to(torch.float32),grad_mask.squeeze())
        return loss_grad

        pred_grad_=pred_grad
        pred_poolmax=self.pool(pred_grad_)
        pred_poolmin=-self.pool(-pred_grad_) 
        pred_grad_=pred_grad_[:,:,self.margin:-self.margin,self.margin:-self.margin]
        pred_par=(pred_grad_-pred_poolmin)/(pred_poolmax-pred_poolmin)
        pred_par=pred_par.to(torch.float32)

        

        tgt_poolmax=self.pool(target_grad)
        tgt_poolmin=-self.pool(-target_grad)
        tgt_grad_=target_grad[:,:,self.margin:-self.margin,self.margin:-self.margin]
        tgt_par=(tgt_grad_-tgt_poolmin)/(tgt_poolmax-tgt_poolmin)
        tgt_par=tgt_par.to(torch.float32)

        grad_mask_=grad_mask[:,:,self.margin:-self.margin,self.margin:-self.margin]
        grad_mask_=grad_mask_>0
        grad_mask_=grad_mask_*(tgt_par>0.5)

        return loss_grad

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

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.__data_loss = TrimmedMAELoss
        self.__regularization_loss = GradientLoss(scales=4, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        total = self.__data_loss(self.__prediction_ssi, target_, mask)

        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class TrimmedProcrustesLoss_noalign(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss_noalign, self).__init__()

        self.__data_loss = TrimmedMAELoss
        self.__regularization_loss = GradientLoss(scales=4, reduction=reduction)
        self.__alpha = alpha
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        self.__prediction_ssi = prediction
        target_ = target

        total = self.__data_loss(self.__prediction_ssi, target_, mask)

        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class TrimmedProcrustesLoss_multimask(nn.Module):
    def __init__(self):
        super(TrimmedProcrustesLoss_multimask, self).__init__()

        self.__data_loss = TrimmedMAELoss
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        nature_img_num,mask_num=prediction.shape[0],mask.shape[1]
        total = torch.zeros(nature_img_num*mask_num)
        lossindex=0
        for img_i in range(nature_img_num):
            prediction_i = prediction[img_i].unsqueeze(0)
            target_i = target[img_i].unsqueeze(0)
            mask_i = mask[img_i].unsqueeze(0)
            for mask_index in range(mask_num):
                mask_patch = mask_i[:,mask_index,:,:]

                percentile = 10
                flattened_tensor = target_i.squeeze(1)[mask_patch].view(-1)
                top_percentile_threshold = torch.kthvalue(flattened_tensor, int(len(flattened_tensor) * (1 - percentile / 100)))
                mask_thr = (target_i >= top_percentile_threshold.values)
                dil_size = 3
                mask_thr = F.pad(mask_thr, (dil_size//2, dil_size//2, dil_size//2, dil_size//2), mode='constant', value=0)
                mask_thr = F.max_pool2d(mask_thr.float(), kernel_size=dil_size, stride=1).squeeze(0).byte()
                mask_patch = (mask_thr > 0) * (mask_patch > 0)


                prediction_ssi = normalize_prediction_robust(prediction_i.squeeze(1), mask_patch)
                target_ssi = normalize_prediction_robust(target_i.squeeze(1), mask_patch)
                total[lossindex]=(self.__data_loss(prediction_ssi, target_ssi, mask_patch, trim=0.5)) 
                lossindex+=1
        total = torch.mean(total)
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class gradssiLoss_randommask(nn.Module):
    def __init__(self, trimed=0.5, region_size=448*2, random_seed=42):
        super(gradssiLoss_randommask, self).__init__()
        self.__data_loss = TrimmedMAELoss
        self.__prediction_ssi = None
        self.trimed, self.region_size, self.random_seed = trimed, region_size, random_seed
    
    def random_region_masks(self, tensor, region_size, random_seed=None):
        torch.manual_seed(random_seed)
        num_channels, height, width = tensor.shape
        masks = torch.zeros_like(tensor, dtype=torch.bool)
        for c in range(num_channels):
            start_h = torch.randint(0, height - region_size + 1, (1,))
            start_w = torch.randint(0, width - region_size + 1, (1,))
            masks[c, start_h:start_h+region_size, start_w:start_w+region_size] = True
        return masks


    def forward(self, prediction, target, mask_edge):
        mask_random = self.random_region_masks(target, self.region_size, self.random_seed)
        mask_final = mask_edge*mask_random
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask_final)
        target_ = normalize_prediction_robust(target, mask_final)
        total = self.__data_loss(self.__prediction_ssi, target_, mask_final, self.trimed)
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi
    prediction_ssi = property(__get_prediction_ssi)

if __name__ == '__main__':
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))

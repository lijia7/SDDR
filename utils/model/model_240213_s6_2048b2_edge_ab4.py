import torch,colorama
colorama.init(autoreset=True)
import copy,tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import func

from torch.nn.functional import interpolate as resize
from utils.backbone import *
from utils.loss import ScaleAndShiftInvariantLoss,GradL1Loss,SILogLoss,\
    TrimmedProcrustesLoss,mygradloss,ScaleAndShiftInvariantLoss_trimed,TrimmedProcrustesLoss_noalign,\
    TrimmedProcrustesLoss_multimask,gradssiLoss_randommask
from utils import loss_my
    

import utils.Resnet as Resnet
import utils.submodule as submodule

from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.cvt import CVT

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bicubic")

def normalize_input_ssi(target, mask):
    ssum = torch.sum(mask, (1, 2, 3))
    valid = ssum > 0
    m = torch.zeros_like(ssum,dtype=torch.float32)
    s = torch.ones_like(ssum,dtype=torch.float32)

    # print(m.dtype,mask.dtype,target.dtype)
    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1, 1)
    sq = torch.sum(mask * target.abs(), (1, 2, 3))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    return target / (s.view(-1, 1, 1, 1))

class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()
        self.upsize = 7
        channel_list = [8,16,32,32,64,160,256] # mitb0
        self.channel_list = channel_list
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.cvt = nn.Sequential()

        self.backbone = mit_b0(img_size_=1024)
        self.backbone.init_weights('/data/wangyiran/work2/mymodels/mit_pretrain/mit_b0.pth')
        
        self.grad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=True)
        old_conv = self.backbone.patch_embed1.proj
        
        new_conv = nn.Conv2d(16, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        self.backbone.patch_embed1.proj = new_conv

        for index in range(len(channel_list)):
            vise_ind = len(channel_list) - index - 1
            # print('in/out ch',channel_list[vise_ind],channel_list[vise_ind-1])
            if index < len(channel_list)-2:
                if index>0:
                    layer = nn.Sequential(nn.GroupNorm(channel_list[vise_ind]//8,channel_list[vise_ind]),  \
                            nn.Conv2d(in_channels=channel_list[vise_ind], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.Sigmoid())
                    self.decoder.add_module('layer_{}_f'.format(index), layer)

                layer = nn.Sequential(nn.GroupNorm(channel_list[vise_ind]//8,channel_list[vise_ind]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), nn.Conv2d(in_channels=channel_list[vise_ind], out_channels=channel_list[vise_ind-1], kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.LeakyReLU())
                # layer = submodule.FTB_2brch_5_1(inchannels=channel_list[vise_ind],midchannels=channel_list[vise_ind],outchannels=channel_list[vise_ind-1])
                self.decoder.add_module('layer_{}'.format(index), layer)
            else:
                layer = nn.Sequential(nn.GroupNorm(channel_list[vise_ind]//8,channel_list[vise_ind]),  \
                            nn.Conv2d(in_channels=channel_list[1], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.Sigmoid())
                self.decoder.add_module('layer_{}_f'.format(index), layer)

                layer = nn.Sequential(nn.GroupNorm(1,channel_list[1]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), nn.Conv2d(in_channels=channel_list[1], out_channels=channel_list[0], kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.LeakyReLU())
                # layer = submodule.FTB_2brch_5_1(inchannels=channel_list[1],midchannels=channel_list[1],outchannels=channel_list[0])
                self.decoder.add_module('layer_{}'.format(index), layer)

                layer = nn.Sequential(nn.GroupNorm(1, channel_list[0]), nn.Conv2d(in_channels=channel_list[0], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.Sigmoid())
                self.decoder.add_module('layer_{}_f'.format(index+1), layer)
                
                layer = nn.Sequential(nn.GroupNorm(1, channel_list[0]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), nn.Conv2d(in_channels=channel_list[0], out_channels=1, kernel_size=5, padding=2, stride=1, padding_mode='replicate', bias=True))
                # layer = submodule.FTB_2brch_5_1(inchannels=channel_list[0],midchannels=channel_list[0],outchannels=1,kernelsize=5)
                self.decoder.add_module('layer_{}'.format(index+1), layer)
                break


        self.mid_out_half = nn.Sequential(nn.GroupNorm(1,channel_list[1]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[1], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), \
                        )
                        #nn.ReLU())
        self.mid_out_quart = nn.Sequential(nn.GroupNorm(2,channel_list[2]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[2], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), \
                        )
                        #nn.ReLU())

        self.en_layer1 = Resnet.BasicBlock_ini(inplanes=1, planes=channel_list[0], stride=2, downsample=nn.Sequential(nn.Conv2d(1, channel_list[0],kernel_size=1, stride=2, bias=True),nn.GroupNorm(channel_list[0]//8,channel_list[0])))
        self.en_layer2 = Resnet.BasicBlock_ini(inplanes=channel_list[0], planes=channel_list[1], stride=2, downsample=nn.Sequential(nn.Conv2d(channel_list[0], channel_list[1],kernel_size=1, stride=2, bias=True),nn.GroupNorm(channel_list[1]//8,channel_list[1])))
        self.en_layer3 = Resnet.BasicBlock_ini(inplanes=channel_list[1], planes=channel_list[2], stride=2, downsample=nn.Sequential(nn.Conv2d(channel_list[1], channel_list[2],kernel_size=1, stride=2, bias=True),nn.GroupNorm(channel_list[2]//8,channel_list[2])))
        self.encoder.add_module('layer_1', self.en_layer1)
        self.encoder.add_module('layer_2', self.en_layer2)
        self.encoder.add_module('layer_3', self.en_layer3)

        self.cross = CVT(input_channel=channel_list[-1], downsample_ratio=1, iter_num=8)

        self.cvt.add_module('cvt_1', self.cross)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.init_weight()


    def init_weight(self):
        for m in self.modules():
            if m in self.backbone.modules():
                continue
            # print(m)
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, x, y):
        y0 = self.grad(y)
        feature_low = []
        feature_high = []
        for layer in self.encoder.named_children():
            if layer[0] == 'layer_2':
                x_2 = layer[1](feature_low[-1])
                y_2 = layer[1](feature_high[-1])
                feature_low.append(x_2)
                feature_high.append(y_2)
            elif layer[0] == 'layer_1':
                feature_low.append(layer[1](x))
                feature_high.append(layer[1](y0))
            else:
                feature_low.append(layer[1](feature_low[-1]))
                feature_high.append(layer[1](feature_high[-1]))

        feature_low += self.backbone(x_2)
        feature_high += self.backbone(y_2)


        feature_cvt=torch.stack([feature_low[-1],feature_high[-1]], dim=1)
        feature_cvt=self.cross(feature_cvt)
        feature_low_cvt=feature_cvt[:,0,:,:,:]
        feature_high_cvt=feature_cvt[:,1,:,:,:]
        feature_low[-1] = feature_low_cvt + feature_low[-1]
        feature_high[-1] = feature_high_cvt + feature_high[-1]


        feature_low.reverse()
        feature_high.reverse()



        index = 0
        for layer in self.decoder.named_children():
            if layer[0] == 'layer_0':
                result = layer[1](feature_low[index] + feature_high[index])
            else:
                if layer[0] == f'layer_{self.upsize-2}':
                    out_half = self.mid_out_half(feature_lh + result)
                    feature_mask_quart = feature_mask
                    result = layer[1](result+feature_lh)
                    index += 1
                    continue
                elif layer[0] == f'layer_{self.upsize-3}':
                    out_quart = self.mid_out_quart(feature_lh + result)
                    result = layer[1](result+feature_lh)
                    index += 1
                    continue
                elif layer[0] == f'layer_{self.upsize-1}':
                    feature_mask_half = feature_mask
                    result = layer[1](result+feature_lh)
                    index += 1
                    continue
                if layer[0][-1] == 'f':
                    feature_mask = layer[1](result)
                    feature_lh = feature_mask*feature_low[index]+(1-feature_mask)*feature_high[index] 
                    continue 
                result = layer[1](result + feature_lh)
            index += 1
        
        return result, out_half, out_quart, feature_mask_quart, feature_mask_half

class Gradient_FusionModel_FFM(nn.Module):
    def __init__(self, args, log_path=None, dict_path=None):
        self.losspara=[float(i) for i in args.losspara]
        print("ini loss Coefficient with",self.losspara)
        device = torch.device('cuda')
        super(Gradient_FusionModel_FFM, self).__init__()
        self.Fuse = FuseNet()
        self.boost_cat = False
        self.pool = nn.AdaptiveAvgPool2d((2*2**10, 2*2**10))
        self.down_sample = nn.AvgPool2d(2, 2)

        # self.ssiloss = ScaleAndShiftInvariantLoss_trimed()
        self.gradloss = TrimmedProcrustesLoss(alpha=0)
        self.gradloss_multimask = TrimmedProcrustesLoss_multimask()
        self.gradloss_random = gradssiLoss_randommask(trimed=0.2, region_size=1024, random_seed=42)
        self.edge_xk = loss_my.EdgeguidedRankingLoss(point_pairs=3000*4)
        self.midasloss = TrimmedProcrustesLoss()
        print("Gradient_FusionModel_FFM stage6 initialized")
        self.sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_x=self.sobel_x.to(device)
        self.sobel_y=self.sobel_y.to(device)

        if log_path is not None:
            self.log_path = log_path
            self.record_dep = np.array([])
            self.record_rank = np.array([])
            self.total_step = 0

        if dict_path is not None:
            depth_dict = torch.load(dict_path)
            self.Fuse.load_state_dict(depth_dict['net'])


    def cal_loss(self, pred_list, inputlist, gt, mask_ssi, gradlist, featuremasklist, ranklist):
        # print(pred_list[0].shape,pred_list[1].shape,pred_list[2].shape,guided.shape,low_dep.shape,high_dep.shape)
        grad, mask_gradloss, fromdataset, mask_edge, mask_flat = gradlist
        lowinput, highinput = inputlist
        feature_quart_mask, feature_half_mask = featuremasklist
        scalessi_para, featuremask_para, grad_para, rankpara = self.losspara
        loss_ssi, loss_ssi_ = 0, 0
        loss_grad = 0
        loss_feature = 0
        loss_mygrad, loss_grad_nature, loss_grad_syn, loss_grad_edge, loss_grad_local = 0,0,0,0,0
        pred_ret = pred_list[0]
        gt_pseudo, img_raw = ranklist

        is_naturescene = torch.tensor([item == 'hr-wsi' for item in fromdataset])

        ermap=None
        if scalessi_para>0:
            for index in range(len(pred_list)):
            # for index in range(1):
                pred = pred_list[index]
                gt_ = torch.nn.functional.interpolate(gt,(pred.shape[2], pred.shape[3]),mode='nearest')
                mask_ = torch.nn.functional.interpolate(mask_ssi.to(torch.uint8),(pred.shape[2], pred.shape[3]),mode='nearest')
                mask_ = mask_>0
                # print("midasloss",pred.squeeze().shape,gt.squeeze().shape,mask_ssi.squeeze().shape,mask_ssi.squeeze().min(),mask_ssi.squeeze().max())
                if mask_ssi.squeeze().max()==False:
                    loss_ssi_ = 0
                else:
                    loss_ssi_ = scalessi_para*self.midasloss(pred.squeeze(), gt_.squeeze(), mask_.squeeze())
                tqdm.tqdm.write(f"loss_ssi_ {'%.4f'%loss_ssi_} // pred shape minmax {pred.shape[-2:]} {'%.4f'%pred.min()} {'%.4f'%pred.max()} // gt{gt.shape[-2:]} {'%.4f'%gt.min()} {'%.4f'%gt.max()}")
                loss_ssi = loss_ssi + loss_ssi_

        s_low, t_low = func.compute_scale_and_shift(pred_ret.squeeze(),lowinput.squeeze(),torch.ones_like(pred_ret.squeeze(),dtype=torch.bool))
        pred_alignlow = s_low.view(-1, 1, 1, 1)*pred_ret + t_low.view(-1, 1, 1, 1)
        lowinputmax, lowinputmin = torch.max(lowinput.view(lowinput.shape[0],-1), dim=-1).values, torch.min(lowinput.view(lowinput.shape[0],-1),dim=-1).values
        pred_alignlowmax, pred_alignlowmin = torch.max(pred_alignlow.view(pred_alignlow.shape[0],-1), dim=-1).values, torch.min(pred_alignlow.view(pred_alignlow.shape[0],-1),dim=-1).values
        
        loss_con_min, loss_con_max = func.error_percent(pred_alignlowmin,lowinputmin), func.error_percent(pred_alignlowmax,lowinputmax)
        tqdm.tqdm.write(colorama.Fore.YELLOW+f"min/max%: {np.round(loss_con_min.tolist(),2)} {np.round(loss_con_max.tolist(),2)}")
        
        
        # !FeatureMaskLoss
        if featuremask_para>0:
            grad_thr_f=0.9-0.05
            mask_loss_edge_,mask_loss_flat_ = 0,0
            for feature_mask in [feature_half_mask,feature_quart_mask]:
                feature_mask_,_ = torch.sort(feature_mask.reshape(-1),dim=-1)  # asend
                big_thr = feature_mask_[int(feature_mask_.shape[0]*(1-grad_thr_f))]
                sml_thr = feature_mask_[int(feature_mask_.shape[0]*grad_thr_f)] 
                grad_mask_=torch.nn.functional.interpolate(mask_edge.to(torch.float32),feature_mask.shape[-2:],mode='nearest')
                grad_mask_=grad_mask_>0
                mask_loss_edge=feature_mask[grad_mask_]
                mask_loss_edge=mask_loss_edge[mask_loss_edge>big_thr]
                grad_mask_=torch.nn.functional.interpolate(mask_flat.to(torch.float32),feature_mask.shape[-2:],mode='nearest')
                grad_mask_=grad_mask_>0
                mask_loss_flat=feature_mask[grad_mask_]
                mask_loss_flat=1-mask_loss_flat[mask_loss_flat<sml_thr]
                if mask_loss_edge.shape[0] != 0:
                    mask_loss_edge_=torch.mean(torch.abs(mask_loss_edge)) #+torch.mean(torch.abs(feature_loss_small))
                else:
                    mask_loss_edge_=0
                if mask_loss_flat.shape[0] != 0:
                    mask_loss_flat_=torch.mean(torch.abs(mask_loss_flat)) #+torch.mean(torch.abs(feature_loss_small))
                else:
                    mask_loss_flat_=0
                loss_feature = loss_feature + featuremask_para*(mask_loss_edge_ + mask_loss_flat_)
                tqdm.tqdm.write(f"feature mask{feature_mask.shape[-2:]} // mask_loss_edge {'%.4f'%mask_loss_edge_} // mask_loss_flat {'%.4f'%mask_loss_flat_} \
                                // FeatureMask edge{'%.2f'%feature_half_mask.min()}({'%.2f'%big_thr}) /flat{'%.2f'%feature_half_mask.max()}({'%.2f'%sml_thr})")

        # MyGradLoss
        if grad_para>0:
            pred = pred_list[0]
            pred_grad_x = F.conv2d(pred, self.sobel_x, stride=1, padding=1,)
            pred_grad_y = F.conv2d(pred, self.sobel_y, stride=1, padding=1,)
            pred_grad=torch.abs(pred_grad_x)+torch.abs(pred_grad_y)
            loss_grad_local = grad_para * self.gradloss(pred_grad.squeeze(), grad.squeeze(), mask_edge.squeeze())
        # RankLoss
        if rankpara>0:
            loss_grad_edge = rankpara * self.edge_xk(pred_ret, gt_pseudo, img_raw) # loss_grad_edge = 0 
        loss_mygrad = grad_para*(loss_grad_edge+loss_grad_local)
        loss_grad = loss_feature+loss_mygrad
        tqdm.tqdm.write(f"loss_ssi {'%.4f'%loss_ssi} // loss_grad{'%.4f'%loss_grad}=loss_feature{'%.4f'%loss_feature}+loss_mygrad{'%.4f'%loss_mygrad}")
        tqdm.tqdm.write(f"loss_mygrad {'%.4f'%loss_mygrad}=loss_grad_edge{'%.4f'%loss_grad_edge}+loss_grad_local{'%.4f'%loss_grad_local}")
        return loss_ssi, loss_grad, ermap


    def predict(self, low, high):
        Fusion, out_half, out_quart, out_half_mask, out_mask = self.Fuse(low, high)
        return Fusion, out_half, out_quart, out_half_mask, out_mask


    def inference(self, low_dep, high_dep):
        low_dep = (low_dep - low_dep.min()) / (low_dep.max() - low_dep.min())
        high_dep = (high_dep - high_dep.min()) / (high_dep.max() - high_dep.min())
        Fusion, out_half, out_quart, out_half_mask, out_mask = self.predict(low_dep, high_dep)
        return out_half_mask, out_mask, Fusion


    def cret(self, low, high, gt, mask_ssi, gradlist, ranklist):
        axis = np.random.randint(0, 2) # 数据增强
        low_ssi, high_ssi = normalize_input_ssi(low,torch.ones_like(low,dtype=torch.bool)), normalize_input_ssi(high,torch.ones_like(high,dtype=torch.bool))
        Fusion, out_half, out_quart, feature_half_mask, feature_mask = self.predict(low, high) # featuremasklist
        loss_low, loss_rank, ermap = self.cal_loss([Fusion, out_half, out_quart], [low_ssi, high_ssi],\
                                                   gt, mask_ssi, gradlist, [feature_half_mask, feature_mask], ranklist)
        self.total_step += 1
        return loss_low + loss_rank, Fusion, ermap
import os
import re
import csv
import cv2
import glob,sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path.append('/data1/lijiaqi/23w1/repo')
# from loss import *

def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)
    return calib


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian
        dispariy = np.fromfile(pfm_file, endian + 'f')
    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib_file_path=None):
    dispariy, [shape,scale] = read_pfm(pfm_file_path)
    if calib_file_path is None:
        raise Exception("Loss calibration information.")
    else:
        calib = read_calib(calib_file_path)
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])
        depth_map = fx*base_line / (dispariy / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).astype('uint8')

        dispariy = np.reshape(dispariy, newshape=shape)
        dispariy = np.flipud(dispariy)
        return dispariy, depth_map

def compute_global_errors(gt, pred):  
    #compute global relative errors
    thresh = np.maximum((gt / pred), (pred / gt))
    thr1 = (thresh < 1.25   ).mean()
    thr2 = (thresh < 1.25 ** 2).mean()
    thr3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    sq_rel = np.mean(((gt - pred)**2) / gt)
    abs_rel = np.mean((np.abs(gt - pred)) / gt)
    return abs_rel, sq_rel, rmse, log10, thr1, thr2, thr3

if __name__ == "__main__":
    imgindex = 0
    gtdir="demo_input/Middlebury2021/GTData"
    preddir,pred_is_disp="demo_output/Middlebury2021/LeRes_Fusion",False
    imglist=sorted(os.listdir(preddir))
    abs_rel,sq_rel,rmse,log10,thr1,thr2,thr3 = np.zeros(len(imglist)),np.zeros(len(imglist)),np.zeros(len(imglist)),np.zeros(len(imglist)),np.zeros(len(imglist)),np.zeros(len(imglist)),np.zeros(len(imglist))
    for imgname in imglist:
        scenename = imgname.split('_')[0]
        

        pfm_file_path = os.path.join(gtdir,scenename,f"disp{imgindex%2}.pfm")
        calib_file_path = os.path.join(gtdir,scenename,f"calib.txt")

        gt_dispariy, gt_depth = create_depth_map(pfm_file_path,calib_file_path)
        gt_depth = gt_depth.squeeze()
        gt_mask = gt_depth>0
        if pred_is_disp:
            gt_depth2disp = 1.0/gt_depth
            gt_depth2disp[~gt_mask] = 0

        pred = cv2.imread(os.path.join(preddir,imgname[:-4]+'.png'),-1)
        pred = cv2.resize(pred,(gt_depth.shape[1],gt_depth.shape[0]))

        pred_flat = pred[gt_mask]
        if pred_is_disp:
            gt_flat = gt_depth2disp[gt_mask]
        else:
            gt_flat = gt_depth[gt_mask]
        coefficients = np.polyfit(pred_flat, gt_flat, deg=1)
        pred_aligned = np.polyval(coefficients, pred)
        if pred_is_disp:
            pred_aligned[pred_aligned<0]=0
            pred_aligned=1.0/(pred_aligned)
        


        
        print(imgindex,gt_depth.shape,gt_depth[gt_mask].min(),gt_depth[gt_mask].max(),pred_aligned.min(),pred_aligned.max())
        print('thr1   = ',  np.nanmean(thr1[thr1>0]))
        abs_rel[imgindex],sq_rel[imgindex],rmse[imgindex],log10[imgindex],thr1[imgindex],thr2[imgindex],thr3[imgindex]=compute_global_errors(gt_depth[gt_mask],pred_aligned[gt_mask])
        imgindex +=1
    
    print('Results:')
    print('abs_rel = ',  np.nanmean(abs_rel[abs_rel>0]))
    print('sq_rel = ',  np.nanmean(sq_rel[sq_rel>0]))
    print('rms    = ',  np.nanmean(rmse[rmse>0]))
    print('log10  = ',  np.nanmean(log10[log10>0]))
    print('thr1   = ',  np.nanmean(thr1[thr1>0]))
    print('thr2   = ',  np.nanmean(thr2[thr2>0]))
    print('thr3   = ',  np.nanmean(thr3[thr3>0]))

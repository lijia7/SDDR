import os,argparse,cv2
import time,glob
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.func import norm_0_1
from torchvision.transforms import Compose
from utils.model.model_240213_s6_2048b2_edge_ab4 import Gradient_FusionModel_FFM
import utils.utils_2stage as utils_2stage

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgsize=2048
torch.manual_seed(42)
torch.multiprocessing.set_start_method('spawn')


parser = argparse.ArgumentParser()
parser.add_argument('--losspara', nargs='+', help='Coefficient for SSI+Grad(midas)/FeatureMask/LocalGrad Loss', required=False, default=[1.0,0,0])
parser.add_argument('--weight', required=False, default="checkpoints/model_dict_1_5600.pt", help='SDDR Model Weight')
parser.add_argument('--rgb', required=False, default="demo_input/Middlebury2021/rgb", help='path to rgb folder')
parser.add_argument('--output', required=False, default="demo_output/Middlebury2021/LeRes_Fusion", help='path to output folder')
parser.add_argument('--low', required=False, default="demo_input/Middlebury2021/LeRes/low_448", help='path to base prediction folder(low resolution)')
parser.add_argument('--high', required=False, default="demo_input/Middlebury2021/LeRes/high_1920", help='path to base prediction folder(high resolution)')
parser.add_argument('--twostage', default=False, help='use two stage refinement', action='store_true')
parser.add_argument('--pred_model', default='LeRes50', help='model type: LeRes50, SGR ,MiDaS')
args = parser.parse_args()  




if not args.twostage:
    Fuse_model = Gradient_FusionModel_FFM(args, dict_path=args.weight)
    Fuse_model.to(device)
    Fuse_model = Fuse_model.eval()
    os.makedirs(args.output, exist_ok=True)
    imgnamelist = sorted(os.listdir(args.low))
    transform_h = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(imgsize, imgsize),interpolation=transforms.InterpolationMode.NEAREST)]) 
    for imgname in imgnamelist:
        ext='.png'
        print(os.path.join(args.rgb,imgname[:-4]+ext),os.path.exists(os.path.join(args.rgb,imgname[:-4]+ext)))
        if os.path.exists(os.path.join(args.rgb,imgname[:-4]+ext)) == False:
            continue
        rgb = cv2.imread(os.path.join(args.rgb,imgname[:-4]+ext)) # h w c
        tgt_w,tgt_h = rgb.shape[1], rgb.shape[0]
        low_dep = cv2.imread(os.path.join(args.low,imgname),-1).astype(np.float32)
        if os.path.exists(os.path.join(args.high,imgname)) == False:
            raise RuntimeError('high no exist')
        high_dep = cv2.imread(os.path.join(args.high,imgname),-1).astype(np.float32)
        low_input, high_input = transform_h(low_dep).unsqueeze(0), transform_h(high_dep).unsqueeze(0)
        low_input = (low_input - low_input.min())/(low_input.max() - low_input.min())
        high_input = (high_input - high_input.min())/(high_input.max() - high_input.min())

        out_half_mask, out_mask, fusion = Fuse_model.inference(low_input.to(device), high_input.to(device))

        fusion = fusion.squeeze().detach().cpu().numpy()
        fusion = cv2.resize(fusion, (tgt_w, tgt_h)) # w h
        cv2.imwrite(os.path.join(args.output,imgname),norm_0_1(fusion))
else:
    Depth_model = utils_2stage.DepthModelSelect('LeRes50',device)
    Fuse_model = Gradient_FusionModel_FFM(args,dict_path=args.weight)
    Fuse_model.to(device)
    Depth_model.to(device)
    Fuse_model = Fuse_model.eval()
    Depth_model = Depth_model.eval()
    utils_2stage.ParamidBoostEval_Leres(Depth_model,Fuse_model,args)
    
    

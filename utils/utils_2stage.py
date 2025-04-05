import torch,cv2,os,colorama,sys,tqdm,glob
# colorama.init(autoreset=True)
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torch.autograd import Variable

import utils.func as func

from thirdrepo.MiDaS.midas_net import MidasNet
from thirdrepo.LeRes.multi_depth_model_woauxi import strip_prefix_if_present, RelDepthModel
from thirdrepo.SGR import DepthNet as SGRnet

def DepthModelSelect(pred_model,device):
    size = 224
    transform_low,transform_high=None,None
    if pred_model == 'LeRes50':
        Depth_model = RelDepthModel(backbone='resnet50')
        depth_dict = './thirdrepo/LeRes/res50.pth'
        depth_dict = torch.load(depth_dict)
        Depth_model.load_state_dict(strip_prefix_if_present(depth_dict['depth_model'], "module."), strict=True)
        model_flag = 1
    
    elif pred_model == 'LeRes101':
        print("select LeRes101 as depth predictor")
        Depth_model = RelDepthModel(backbone='resnext101')
        depth_dict = './thirdrepo/LeRes/res101.pth'
        depth_dict = torch.load(depth_dict)
        Depth_model.load_state_dict(strip_prefix_if_present(depth_dict['depth_model'], "module."), strict=True)
        model_flag = 1

    elif pred_model == 'SGR':
        Depth_model = SGRnet.DepthNet()
        if device == torch.device("cuda"):
            Depth_model = torch.nn.DataParallel(Depth_model, device_ids=[0]).cuda()
        else:
            print('sgr model can not run correctly without cpu')
            exit()
        depth_dict = torch.load('./thirdrepo/SGR/model.pth.tar')
        Depth_model.load_state_dict(depth_dict['state_dict'])
        model_flag = 2
    
    elif pred_model == 'midas':
        Depth_model = MidasNet('./thirdrepo/MiDaS/model.pt', non_negative=True)
        model_flag = 3
        size = 192

    else:
        print('no such model')
        exit()
    return Depth_model

def TransformImgtoModel_2scale(img,pred_model,device):
    if pred_model == 'LeRes50':
        img = img.astype('float32')/255.0
        img = img[:, :, ::-1]
        low_img = func.img2Tensor(img, 1, int(448*1.0)).to(device)
        high_img = func.img2Tensor(img, 1, 1920).to(device)
    elif pred_model == 'SGR':
        img = img.astype('float32')/255.0
        img = img[:, :, ::-1]
        low_img = func.img2Tensor(img, 1, int(448*1.0)).to(device)
        high_img = func.img2Tensor(img, 1, int(448*2.0)).to(device)
    elif pred_model == 'midas':
        img = img.astype('float32')/255.0
        img = img[:, :, ::-1]
        print("infer with midas 224 2 6",img.min(),img.max())
        low_img = func.img2Tensor(img, 1, int(192*2.0)).to(device)
        high_img = func.img2Tensor(img, 1, int(192*4.0)).to(device) # S6000
    return low_img, high_img

def DepthModelInfer_DepthFuse(Depth_model,low_img,high_img,gtdepth,pred_model,\
                              logweight,step_index,img_index,device):
    print(colorama.Fore.BLUE+f"infer with {pred_model}")
    with torch.no_grad():
        if pred_model[:5] == 'LeRes': # LeRes50 2
            low_dep = Depth_model.inference(low_img,norm=False)
            high_dep = Depth_model.inference(high_img,norm=False)
            low_dep = (low_dep - low_dep.min())/(low_dep.max() - low_dep.min())
            high_dep = (high_dep - high_dep.min())/(high_dep.max() - high_dep.min())

        elif pred_model == 'SGR': # SGR 2
            low_dep = Depth_model.forward(low_img)
            high_dep = Depth_model.forward(high_img)
            low_dep = (low_dep - low_dep.min())/(low_dep.max() - low_dep.min())
            high_dep = (high_dep - high_dep.min())/(high_dep.max() - high_dep.min())

        elif pred_model == 'midas': #midas
            low_dep = Depth_model.forward(low_img.to(device)).unsqueeze(0)
            high_dep = Depth_model.forward(high_img.to(device)).unsqueeze(0)
            img_vis=low_img.squeeze().squeeze().detach().cpu().numpy().transpose((1,2,0))[:,:,::-1]
            low_dep = (low_dep - low_dep.min())/(low_dep.max() - low_dep.min())
            high_dep = (high_dep - high_dep.min())/(high_dep.max() - high_dep.min())
    return low_dep, high_dep


#-------------------Boosting-----------------------------------
import skimage.measure
from thirdrepo.boost_utils import ImageandPatchs, ImageDataset, generatemask, getGF_fromintegral, \
                        calculateprocessingres, rgb2gray, applyGridpatch, myImageandPatchs
boost_info_color=colorama.Fore.YELLOW
whole_size_threshold = 3000 # R_max from the paper
GPU_threshold = 2048 # Limit for the GPU (NVIDIA RTX 6000), can be adjusted 


def TransformImgtoModel(img,msize,pred_model,device):
    if pred_model == 'dpt':
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = Compose(
            [Resize(
                msize, msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),])
        img_ = img.astype('float32')/255.0
        img_ = transform({"image": img_})["image"]
    elif pred_model == 'newcrfs':
        img_ = img.astype('float32')/255.0
        img_ = cv2.resize(img_, (msize, msize))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_ = np.expand_dims(img_, axis=0)
        img_ = np.transpose(img_, (0, 3, 1, 2))
        img_ = Variable(normalize(torch.from_numpy(img_)).float()).cuda()
    elif pred_model == 'LeRes50':
        img_ = img.astype('float32')/255.0
        img_ = img_[:,:,::-1]
        img_ = func.img2Tensor(img, 1, msize).to(device)
    elif pred_model == 'midas':
        img_ = func.img2Tensor(img, 1, msize).to(device)
    return img_

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

# Generate a single-input depth estimation
def singleestimate(Depth_model, img, msize, pred_model, device):
    with torch.no_grad():
        if msize > GPU_threshold:
            print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
            msize = GPU_threshold
        if pred_model=='LeRes50':
            img_torch = func.img2Tensor(img.astype(np.float32), 1, int(msize*1.0)).to(device)
            if not hasattr(Depth_model, 'inference'):
                raise RuntimeError('LeRes50 no inference')
                Depth_model_ = DepthModelSelect('LeRes50',device)
                Depth_model_.to(device)
                Depth_model_ = Depth_model_.eval()
                return Depth_model_.inference(img_torch,norm=False)
            return Depth_model.inference(img_torch,norm=False)
        elif pred_model=='LeRes101':
            rgb_c = (img).copy()
            A_resize = cv2.resize(rgb_c, (msize, msize))
            img_torch = scale_torch(A_resize)[None, :, :, :]
            return Depth_model.inference(img_torch,norm=False)
        elif pred_model=='midas':
            rgb_c = (img).copy()
            A_resize = cv2.resize(rgb_c, (msize, msize))
            img_torch = scale_torch(A_resize)[None, :, :, :]
            ret = Depth_model.forward(img_torch.to(device))
            ret = (ret-ret.min())/(ret.max()-ret.min())
            return ret.unsqueeze(0)
        else:
            raise RuntimeError("base single estimate no implementation")

    

# Generate a double-input depth estimation
def doubleestimate(Depth_model, Fuse_model ,img, size1, size2, pix2pixsize, modeltype):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate the low resolution estimation
    if modeltype == 'zoe' and size1>512:
        size1 = 512
    elif modeltype == 'midas' and size1>384:
        size1 = 384
    estimate1 = singleestimate(Depth_model, img, size1, modeltype, device)
    if modeltype=='midas' and size2>384*2:
        size2 = int(384*2)
    estimate2 = singleestimate(Depth_model, img, size2, modeltype, device)
    print(colorama.Fore.BLUE+f"in doubleestimate img: {img.min()}/{img.max()} {size1} {size2} {pix2pixsize} // {estimate1.shape} {estimate2.shape}")
    print(colorama.Fore.BLUE+f"in doubleestimate estimatelow: {estimate1.min()}/{estimate1.max()} // estimatehigh: {estimate2.min()}/{estimate2.max()}")
    transform_h = transforms.Compose([transforms.Resize(size=(pix2pixsize, pix2pixsize))]) 
    low_input, high_input = transform_h(estimate1), transform_h(estimate2)
    _, _, fusion = Fuse_model.inference(low_input, high_input)
    prediction_mapped = fusion

    pred_consis = fusion.clone()
    s_low, t_low = func.compute_scale_and_shift(pred_consis.squeeze(0),low_input.squeeze(0),torch.ones_like(pred_consis.squeeze(0),dtype=torch.bool))
    pred_consis = s_low.view(-1, 1, 1, 1)*pred_consis + t_low.view(-1, 1, 1, 1)
    lowinputmax, lowinputmin = torch.max(low_input.view(low_input.shape[0],-1), dim=-1).values, torch.min(low_input.view(low_input.shape[0],-1),dim=-1).values
    pred_alignlowmax, pred_alignlowmin = torch.max(pred_consis.view(pred_consis.shape[0],-1), dim=-1).values, torch.min(pred_consis.view(pred_consis.shape[0],-1),dim=-1).values
    loss_con_min, loss_con_max = func.error_percent(pred_alignlowmin,lowinputmin), func.error_percent(pred_alignlowmax,lowinputmax)
    tqdm.tqdm.write(colorama.Fore.RED+f"lowinput: {np.round(lowinputmin.detach().cpu(),4)} {np.round(lowinputmax.detach().cpu(),4)}")
    tqdm.tqdm.write(colorama.Fore.RED+f"pred_ssi: {np.round(pred_alignlowmin.detach().cpu(),4)} {np.round(pred_alignlowmax.detach().cpu(),4)}")
    tqdm.tqdm.write(colorama.Fore.RED+f"min/max%: {np.round((loss_con_min/torch.abs(lowinputmin)*100).tolist(),4)} {np.round((loss_con_max/torch.abs(lowinputmax)*100).tolist(),4)}")
    return prediction_mapped,estimate1,estimate2

def torch_norm(input):
    return (input-input.min())/(input.max()-input.min())


def ParamidBoostEval_Leres(Depth_model,Fuse_model,option):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()
    option.savewholeest = True
    option.savepatchs = True
    option.R0, option.R20 = False, False
    option.output_resolution=1 # '0 for results in maximum resolution 1 for resize to input size'
    option.colorize_results = False # option.colorize_results = True
    option.max_res = np.inf

    # For Middlebury2021
    option.data_dir = "demo_input/Middlebury2021/rgb"
    option.step_output_dir = f"demo_output/Middlebury2021/LeRes_2stage/mid21_leres50_step"
    option.output_dir = f"demo_output/Middlebury2021/LeRes_2stage/mid21_final"
    os.makedirs(option.step_output_dir, exist_ok=True) if (not os.path.exists(option.step_output_dir)) else None

    result_dir = option.output_dir
    os.makedirs(result_dir, exist_ok=True)
    if option.savewholeest:
        whole_est_outputpath = option.output_dir + '_wholeimage'
        os.makedirs(whole_est_outputpath, exist_ok=True)
    if option.savepatchs:
        patchped_est_outputpath = option.output_dir + '_patchest'
        os.makedirs(patchped_est_outputpath, exist_ok=True)

    dataset = ImageDataset(option.data_dir, 'test')
    
    # default setting
    scale_threshold = 4  # Allows up-scaling with a scale up to 3
    r_threshold_value = 0.2
    
    # For Leres50
    option.net_receptive_field_size=448
    option.pix2pixsize=1024
    option.patch_netsize=2*option.net_receptive_field_size

    # Go through all images in input directory
    print(boost_info_color+"start processing")
    img_grad_list, rgb_grad_standard = [], 0.4
    import time
    starttime=time.time()
    for image_ind, images in enumerate(dataset):
        print(boost_info_color+'processing image', image_ind, ':', images.name)
        img = images.rgb_image

        img_grad_mean = rgbgrad_patchmin(img.copy())
        img_grad_list.append(img_grad_mean)
        print(boost_info_color+'img_grad_mean:', img_grad_mean)

        input_resolution = img.shape
        whole_image_optimal_size, patch_scale = calculateprocessingres(img, option.net_receptive_field_size,
                                                                        r_threshold_value, scale_threshold,
                                                                        whole_size_threshold)
        if Fuse_model.boost_cat == False:
            base_expand = 1
            whole_image_optimal_size = (int(max(img.shape[:2])*base_expand)//32)*32
        if option.pred_model=='midas':
            whole_image_optimal_size=int(384*2.0)
        elif option.pred_model=='LeRes50':
            whole_image_optimal_size=1024
        whole_estimate,estimate1,estimate2 = doubleestimate(Depth_model, Fuse_model, img, \
                                        option.net_receptive_field_size, whole_image_optimal_size,
                                        option.pix2pixsize, option.pred_model)

        os.makedirs(option.output_dir+'_base',exist_ok=True)
        func.midas_write_depth(os.path.join(option.output_dir+'_base', images.name),cv2.resize(whole_estimate.squeeze().detach().cpu().numpy(),(input_resolution[1], input_resolution[0]),
                                                interpolation=cv2.INTER_CUBIC), bits=2, colored=True)
        os.makedirs(option.output_dir+'_est1',exist_ok=True)
        func.midas_write_depth(os.path.join(option.output_dir+'_est1', images.name),cv2.resize(estimate1.squeeze().detach().cpu().numpy(),(input_resolution[1], input_resolution[0]),
                                                interpolation=cv2.INTER_CUBIC), bits=2, colored=False)
        os.makedirs(option.output_dir+'_est2',exist_ok=True)
        func.midas_write_depth(os.path.join(option.output_dir+'_est2', images.name),cv2.resize(estimate2.squeeze().detach().cpu().numpy(),(input_resolution[1], input_resolution[0]),
                                                interpolation=cv2.INTER_CUBIC), bits=2, colored=False)
        estimate1_ = torch_norm(estimate1).squeeze().detach().cpu().numpy()
        whole_estimate = torch_norm(whole_estimate).squeeze().detach().cpu().numpy()

        if option.R0 or option.R20:
            path = os.path.join(result_dir, images.name)
            if option.output_resolution == 1:
                func.midas_write_depth(path, cv2.resize(whole_estimate, (img.shape[1], img.shape[0]),
                                                         interpolation=cv2.INTER_CUBIC),
                                        bits=2, colored=option.colorize_results)
            else:
                func.midas_write_depth(path, whole_estimate, bits=2, colored=option.colorize_results)
            continue

        # Output double estimation if required
        if option.savewholeest:
            path = os.path.join(whole_est_outputpath, images.name)
            print(path)
            if option.output_resolution == 1:
                whole_estimate_color = func.midas_write_depth(path,
                                        cv2.resize(whole_estimate, (img.shape[1], img.shape[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
                func.midas_write_depth(path+'_est1',
                                        cv2.resize(estimate1.squeeze().detach().cpu().numpy(), (img.shape[1], img.shape[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
                func.midas_write_depth(path+'_est2',
                                        cv2.resize(estimate2.squeeze().detach().cpu().numpy(), (img.shape[1], img.shape[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
            else:
                func.midas_write_depth(path, whole_estimate, bits=2, colored=option.colorize_results)

        img_rawsize = img.copy() # img_rawsize = cv2.resize(img, (myimgsize, myimgsize), interpolation=cv2.INTER_CUBIC)
        
        patchs0_flat_list, patchs1_flat_list = None, None
        use_semanticmask = False # use_semanticmask = True
        if use_semanticmask:
            samenticmasklist=sorted(glob.glob('/data1/lijiaqi/23w1/rgb_segment/mask_final/*'))
            samenticmask = cv2.imread(samenticmasklist[image_ind])
            if samenticmask.shape[-1]==3:
                samenticmask = (samenticmask[:,:,0]+samenticmask[:,:,1]+samenticmask[:,:,2])>0
            else:
                samenticmask = (samenticmask)>0
            if np.sum(samenticmask) != 0:
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(samenticmask.astype(np.uint8), connectivity=8)
                max_area = 0
                for i in range(0, retval):
                    point_centroids_value = samenticmask[int(centroids[i][1]),int(centroids[i][0])]
                    if point_centroids_value == False: # 跳过背景
                        continue
                    if stats[i, cv2.CC_STAT_AREA] > max_area:
                        max_area = stats[i, cv2.CC_STAT_AREA]
                if max_area/samenticmask.size > 0.33:
                    whole_estimate_color = func.midas_write_depth(os.path.join(option.output_dir, images.name), cv2.resize(whole_estimate, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results)
                    continue
                paramidstep_s0, paramidpatchnum_s0, cover_rate_s0 = 0, 2, 0.1
                basepatch_size_s0 = (int(img.shape[0]/(paramidpatchnum_s0-((paramidpatchnum_s0-1)*cover_rate_s0)))+1,\
                                int(img.shape[1]/(paramidpatchnum_s0-((paramidpatchnum_s0-1)*cover_rate_s0)))+1) # (617, 1097)
                imageandpatchs_s0 = myImageandPatchs(option.data_dir, images.name, img, basepatch_size = basepatch_size_s0, cover_rate=cover_rate_s0)
                patchs0_flat_list = imageandpatchs_s0.set_samentic_mask(samenticmask.copy())

                paramidstep_s1, paramidpatchnum_s1, cover_rate_s1 = 1, 3, 0.1
                basepatch_size_s1 = (int(img.shape[0]/(paramidpatchnum_s1-((paramidpatchnum_s1-1)*cover_rate_s1)))+1,\
                                int(img.shape[1]/(paramidpatchnum_s1-((paramidpatchnum_s1-1)*cover_rate_s1)))+1) # (617, 1097)
                imageandpatchs_s1 = myImageandPatchs(option.data_dir, images.name, img, basepatch_size = basepatch_size_s1, cover_rate=cover_rate_s1)
                patchs1_flat_list = imageandpatchs_s1.set_samentic_mask(samenticmask.copy())


        # Paramid Step 0 patch2x2
        paramidstep, paramidpatchnum, cover_rate = 0, 2, 0.2
        basepatch_size = (int(img.shape[0]/(paramidpatchnum-((paramidpatchnum-1)*cover_rate)))+1,\
                          int(img.shape[1]/(paramidpatchnum-((paramidpatchnum-1)*cover_rate)))+1) # (617, 1097)
        imageandpatchs = myImageandPatchs(option.data_dir, images.name, img, basepatch_size = basepatch_size, cover_rate=cover_rate)
        whole_estimate_resized = cv2.resize(whole_estimate, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
        print("whole_estimate_resized",whole_estimate_resized.shape,whole_estimate_resized.min(),whole_estimate_resized.max())
        imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
        imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

        thr_grad_mean, patch_grad_list=imageandpatchs.get_grad_threshold()
        thr_grad_big = sorted(np.array(patch_grad_list)[patch_grad_list>thr_grad_mean])[int(0.5 * np.sum(patch_grad_list>thr_grad_mean))]
        path_ = os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}') + f"_grad"
        func.midas_write_depth(path_, imageandpatchs.estimation_base_grad, bits=2, colored=option.colorize_results)
        expand_factor_flat, expand_factor_mean, expand_factor_edge = 0.8, 1.0, 1.2
        
        imageandpatchs, whole_estimate_color = patchfusion(Depth_model, Fuse_model, imageandpatchs, \
                    [patch_grad_list, thr_grad_big, thr_grad_mean, expand_factor_edge, expand_factor_mean, expand_factor_flat], \
                    whole_estimate_color, mask, mask_org, \
                    option, device, paramidstep, input_resolution, patchs0_flat_list)

        # -------------------------------------------------------
        # Paramid Step 1 patch5x5
        paramidstep, paramidpatchnum, cover_rate = 1, 3, 0.25
        basepatch_size = (int(img.shape[0]/(paramidpatchnum-((paramidpatchnum-1)*cover_rate)))+1,\
                          int(img.shape[1]/(paramidpatchnum-((paramidpatchnum-1)*cover_rate)))+1) # (617, 1097)
        whole_estimate_resized = cv2.resize(imageandpatchs.estimation_updated_image, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
        imageandpatchs = myImageandPatchs(option.data_dir, images.name, img, basepatch_size = basepatch_size, cover_rate=cover_rate)
        imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
        imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())
        
        thr_grad_mean, patch_grad_list=imageandpatchs.get_grad_threshold()
        thr_grad_big = sorted(np.array(patch_grad_list)[patch_grad_list>thr_grad_mean])[int(0.5 * np.sum(patch_grad_list>thr_grad_mean))]
        path_ = os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}') + f"_grad"
        func.midas_write_depth(path_, imageandpatchs.estimation_base_grad, bits=2, colored=option.colorize_results)
        expand_factor_flat, expand_factor_mean, expand_factor_edge = 0.8, 1.0, 1.2
        imageandpatchs, whole_estimate_color = patchfusion(Depth_model, Fuse_model, imageandpatchs, \
                    [patch_grad_list, thr_grad_big, thr_grad_mean, expand_factor_edge, expand_factor_mean, expand_factor_flat], \
                    whole_estimate_color, mask, mask_org, \
                    option, device, paramidstep, input_resolution, patchs1_flat_list)

        func.midas_write_depth(os.path.join(option.output_dir, imageandpatchs.name),cv2.resize(imageandpatchs.estimation_updated_image.squeeze(),(input_resolution[1], input_resolution[0]),
                                                interpolation=cv2.INTER_CUBIC), bits=2, colored=False)
    endtime=time.time()
    print(img_grad_list,(endtime-starttime)/48)

def patchfusion(Depth_model,Fuse_model,imageandpatchs,paralist, \
                whole_estimate_color,mask,mask_org,option,device,paramidstep,input_resolution,semanticlist,save_mid_result=True):
    patch_grad_list,thr_grad_big,thr_grad_mean,expand_factor_edge,expand_factor_mean,expand_factor_flat = paralist
    
    rect_color_num = 20
    rect_colormap = plt.get_cmap('tab20')(range(rect_color_num))
    rect_vis_index, rect_save_index = 0, 0
    if save_mid_result:
        whole_estimate_color_ = whole_estimate_color.copy()
        whole_estimate_gradwhole_box = whole_estimate_color.copy()
        whole_estimate_gradbig_box = whole_estimate_color.copy()
    else:
        whole_estimate_color_,whole_estimate_gradwhole_box,whole_estimate_gradbig_box=None,None,None,
    # Enumerate through all patches, generate their estimations and refining the base estimate.
    for patch_ind in range(len(imageandpatchs)):
        # Get patch information
        patch = imageandpatchs[patch_ind] # patch object
        patch_rgb = patch['patch_rgb'] # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
        rect = patch['rect'] # patch size and location
        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]
        patch_id = patch['id'] # patch ID 即初始划分的序号
        org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
        patchped_est_outputpath = option.output_dir + '_patchest'
        patchsavepath = os.path.join(patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
        print('\t processing patch', patch_ind, '|', rect, "saving to", patchsavepath)

        patch_grad = patch_grad_list[patch_ind] # patch_grad = np.sum(whole_estimate_grad[h1:h2,w1:w2])/((h2-h1)*(w2-w1)) # patch_grad_list.append(patch_grad)
        if patch_grad > thr_grad_big:
            expand_factor = expand_factor_edge
        elif patch_grad > thr_grad_mean:
            expand_factor = expand_factor_mean
        else:
            expand_factor = expand_factor_flat
        transform_h = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(option.pix2pixsize, option.pix2pixsize),interpolation=transforms.InterpolationMode.NEAREST)])
        patch_est_shape = int((max(patch_rgb.shape[:2])*expand_factor+448)//64+1)*32 # expand_factor = 2
        print('\t processing res', patch_est_shape)
        if semanticlist!=None and semanticlist[patch_ind]>0.5:
            print(colorama.Fore.RED+f"\t ({semanticlist[patch_ind]}) weak texture, thus limiting the high infer resolution")
            patch_est_shape = (int(option.net_receptive_field_size*1.1)//32)*32
        print('\t processing grad for patch expand', patch_grad, f"expand_factor:{expand_factor}/{patch_est_shape}/{patch_rgb.shape}")
        
        transform_resize = transforms.Compose([transforms.Resize(size=(option.pix2pixsize, option.pix2pixsize))])
        patch_estimation = singleestimate(Depth_model, patch_rgb, patch_est_shape, option.pred_model, device)
        patch_estimation = transform_resize(patch_estimation).to(device)



        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (option.pix2pixsize, option.pix2pixsize),
                                                interpolation=cv2.INTER_NEAREST)       
        # Output patch estimation if required
        if save_mid_result:
            if option.savepatchs:
                path = os.path.join(patchped_est_outputpath, imageandpatchs.name+f'_step{paramidstep}' + '_{:04}'.format(patch_id))
                func.midas_write_depth(path+'_0whole', patch_whole_estimate_base.squeeze(), bits=2, colored=option.colorize_results)
                func.midas_write_depth(path+'_1patchdouble', patch_estimation.squeeze().detach().cpu().numpy(), bits=2, colored=option.colorize_results)
        patch_whole_estimate_base_ = transform_h(patch_whole_estimate_base).to(device).unsqueeze(1)
        _, _, prediction_mapped = Fuse_model.inference(patch_whole_estimate_base_, patch_estimation)
        if save_mid_result:
            func.midas_write_depth(path+'_2final', prediction_mapped.squeeze().detach().cpu().numpy(), bits=2, colored=option.colorize_results)

        s_low, t_low = func.compute_scale_and_shift(prediction_mapped.squeeze(0),patch_whole_estimate_base_.squeeze(0),torch.ones_like(patch_whole_estimate_base_.squeeze(0),dtype=torch.bool))
        mapped = s_low.view(-1, 1, 1, 1)*prediction_mapped + t_low.view(-1, 1, 1, 1)
        merged = mapped.squeeze().detach().cpu().numpy()
        print(boost_info_color+f"\t{merged.min()}/{merged.max()} // {patch_whole_estimate_base.min()}/{patch_whole_estimate_base.max()}")

        merged = cv2.resize(merged, (org_size[1],org_size[0]), interpolation=cv2.INTER_CUBIC)

        if save_mid_result:
            rect_color = rect_colormap[rect_vis_index%rect_color_num]
            rect_color = (int(rect_color[2]*255),int(rect_color[1]*255),int(rect_color[0]*255))
            rect_vis_index += 1
            cv2.rectangle(whole_estimate_color_, (w1, h1), (w2, h2), rect_color, 15) # thickness
            if patch_grad > thr_grad_mean:
                cv2.rectangle(whole_estimate_gradwhole_box, (w1, h1), (w2, h2), rect_color, 2)
            if patch_grad > thr_grad_big:
                cv2.rectangle(whole_estimate_gradbig_box, (w1, h1), (w2, h2), rect_color, 2)
            if rect_vis_index%rect_color_num == 0 or rect_vis_index == len(imageandpatchs):
                path_ = os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}') + f"_box_{'%03d'%rect_save_index}.png"
                rect_save_index+=1
                if save_mid_result:
                    cv2.imwrite(path_,whole_estimate_color_)
                whole_estimate_color_ = whole_estimate_color.copy()
        

        if expand_factor == 0:
            continue
        if mask.shape != org_size:
            mask = cv2.resize(mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)
        tobemergedto = imageandpatchs.estimation_updated_image
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)
        # break

    if save_mid_result:
        if os.path.exists(option.output_dir+'_color') == False:
            os.makedirs(option.output_dir+'_color')
        func.midas_write_depth(os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}'),cv2.resize(imageandpatchs.estimation_updated_image,(input_resolution[1], input_resolution[0]),interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results)
        path = os.path.join(option.output_dir+'_color', imageandpatchs.name+f'_step{paramidstep}')
        whole_estimate_color = func.midas_write_depth(path,cv2.resize(imageandpatchs.estimation_updated_image, (input_resolution[1], input_resolution[0]),interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results)
    
        cv2.imwrite(os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}') + f"_gradwhole_box.png",whole_estimate_gradwhole_box)
        cv2.imwrite(os.path.join(option.step_output_dir, imageandpatchs.name+f'_step{paramidstep}') + f"_gradbig_box.png",whole_estimate_gradbig_box)
    return imageandpatchs, whole_estimate_color

def rgbgrad_patchmin(rgb_image,patchnum=2):
    num_rows, num_cols = patchnum, patchnum
    patch_height, patch_width = rgb_image.shape[0] // num_rows, rgb_image.shape[1] // num_cols

    average_gradients = []

    for i in range(num_rows):
        for j in range(num_cols):
            patch = rgb_image[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width, :]
            
            gradient_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            average_gradient = np.sum(np.abs(cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)))/patch.size*3.0
            
            average_gradients.append(average_gradient)

    min_average_gradient = np.min(average_gradients)
    return min_average_gradient

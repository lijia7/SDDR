import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from utils.guided_f import guided_filter

def norm_0_1(input):
    output=(input-input.min())/(input.max()-input.min())
    output=(output*65535.0).astype(np.uint16)
    return output

def norm_vis512(input):
    output=(input-input.min())/(input.max()-input.min())
    output=(output*65535.0).astype(np.uint16)
    output=cv2.resize(output,(512,512))
    return output

def midas_write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    # if depth_max>max_val:
    #     print('Warning: Depth being clipped')
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = depth
    #     out [depth > max_val] = max_val
    # else:
    #     out = 0
    print('saving to',path,colored)
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(max_val-out,cv2.COLORMAP_MAGMA)
        cv2.imwrite(path+'.png', out)
    elif bits == 2:
        cv2.imwrite(path+'.png', out.astype("uint16"))

    return out

def compute_scale_and_shift(prediction, target, mask):
    # mask=mask.bool()
    # print("\033[92m in compute_scale_and_shift(prediction/target/mask) \033[0m","\033[94m type \033[0m",prediction.dtype,target.dtype,mask.dtype,\
    #       "\033[94m device \033[0m",prediction.device,target.device,mask.device,\
    #       "\033[94m minmaxw/mask \033[0m",target[mask].min(),target[mask].max(),target[~mask].min(),target[~mask].max()  )
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    target_c = target.clone()
    target_c[~mask] = 0
    b_0 = torch.sum(mask * prediction * target_c, (1, 2))
    b_1 = torch.sum(mask * target_c, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def error_percent(predlist,tgtlist):
    if predlist.shape[0]!=tgtlist.shape[0]:
        raise RuntimeError('len no equal')
    return torch.abs(predlist-tgtlist)
    # for i in range(len(predlist)):
    #     err_p = abs(tgtlist[i]-predlist[i])/abs(tgtlist[i])*100
    #     if err_p<0.05:
    #         err_p = 0
    #     retlist.append(err_p)


def compute_scale_and_shift_consist(prediction, target, mask):
    # mask=mask.bool()
    # print("\033[92m in compute_scale_and_shift(prediction/target/mask) \033[0m","\033[94m type \033[0m",prediction.dtype,target.dtype,mask.dtype,\
    #       "\033[94m device \033[0m",prediction.device,target.device,mask.device,\
    #       "\033[94m minmaxw/mask \033[0m",target[mask].min(),target[mask].max(),target[~mask].min(),target[~mask].max()  )
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    target_c = target.clone()
    target_c[~mask] = 0
    b_0 = torch.sum(mask * prediction * target_c, (1, 2))
    b_1 = torch.sum(mask * target_c, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def shift_scale(pred, gt, mask_=None):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    mask_valid = np.ones_like(gt_flat)
    mask_valid[gt_flat == 0] = 0
    if mask_ is not None:
        mask_ = mask_.flatten()
        mask_valid[mask_ == 0] = 0

    gt_valid = np.array([gt_flat[i] for i in range(len(mask_valid)) if mask_valid[i]])
    pred_valid = np.array([pred_flat[i] for i in range(len(mask_valid)) if mask_valid[i]])

    para_s = np.polyfit(pred_valid, gt_valid, deg=1)
    pred = np.polyval(para_s, pred)
    return pred


# def generate_gf(low_dep, high_dep):
#     r = int(high_dep.shape[0] / 12) - 1
#     enhanced = guided_filter(high_dep, low_dep, r, 1e-12)
#     return enhanced


def visual_crfs(low_dep, high_dep):
    low_dep[:, :, :, :5] = low_dep.min()
    low_dep[:, :, :, -5:] = low_dep.min()
    low_dep[:, :, :5, :] = low_dep.min()
    low_dep[:, :, -5:, :] = low_dep.min()
    high_dep[:, :, :, :15] = high_dep.min()
    high_dep[:, :, :, -15:] = high_dep.min()
    high_dep[:, :, :15, :] = high_dep.min()
    high_dep[:, :, -15:, :] = high_dep.min()
    return low_dep, high_dep


def img2Tensor(img, scale, model_input_size=224):
    # print(scale,model_input_size,scale)
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(model_input_size*scale, model_input_size*scale)),\
        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))])
    tens = transformer(img.copy())
    return tens[None, :, :, :]


def scale_image(img, size, device):
    scale_list = [2, 6]
    # scale_list = [3, 10]
    tensor_list = []
    
    for scale in scale_list:
        tensor = img2Tensor(img, scale, size)
        if device == torch.device("cuda"):
            tensor_list.append(tensor.cuda())
        else:
            tensor_list.append(tensor)
    return tensor_list

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0
    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)
    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)
    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    return target / (s.view(-1, 1, 1))

def save_orig(input_rgb, img_loc, low_dep, pred, model_flag):
    if model_flag >= 5:
        input_rgb = (input_rgb * 255).astype('uint8')
    input_rgb  = cv2.resize(input_rgb, None, fx=0.5, fy=0.5)
    h, w, _ = input_rgb.shape
    low_dep = cv2.resize(low_dep.cpu().detach().numpy().squeeze(), (w, h))
    pred = cv2.resize(pred.cpu().detach().numpy().squeeze(), (w, h))
    low_dep = 255 - (low_dep - low_dep.min()) / (low_dep.max() - low_dep.min()) * 255
    pred = 255 - (pred - pred.min()) / (pred.max() - pred.min()) * 255
    result = np.hstack((low_dep, low_dep, pred))
    plt.imsave(img_loc, result, cmap='inferno')
    result = cv2.imread(img_loc)
    result[:h, :w, :] = input_rgb
    cv2.imwrite(img_loc, result)


# 定义生成二值gradmask的函数
def generate_gradmask(center, size, total_size):
    mask = np.zeros(total_size)
    x, y = center
    half_size = size // 2
    mask[max(x-half_size,0):min(x+half_size+1,total_size[0]), max(y-half_size,0):min(y+half_size+1,total_size[1])] = 1
    return mask.astype(np.uint8)
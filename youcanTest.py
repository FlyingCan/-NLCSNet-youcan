"""
 @Time    : 2023/5/13 14:39
 @Author  : Youcan Xu
 @E-mail  : youcanxv@163.com
 @Project : Code reproduction of Image Compressed Sensing Using Non-local Neural Network
 @File    : test.py
 @Function: test and eval NLCSNet
"""
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
import copy
import cv2
from data_utils import checkdir,psnr
from NL_CSNet import NL_CSNet
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = ArgumentParser(description='ISTA-Net')
parser.add_argument('--epoch_num', type=int, default=200, help='epoch number of model')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--cs_ratio', type=int, default=0.1, help='from {0.01, 0.04, 0.1, 0.25, 0.4, 0.5,0.4}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--model_name', type=str, default='net_epoch_100_0.002229.pth',help='model name')
parser.add_argument('--data_dir', type=str, default='datasets', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set14', help='name of test set')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()
epoch_num = args.epoch_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

model_dir = "./%s/NLCSNet_ratio_%.2f/%s" % (args.model_dir,  cs_ratio, args.model_name)

model = NL_CSNet(args.block_size, args.cs_ratio)
model = model.to(device)
model.load_state_dict(torch.load(model_dir,map_location=device))

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + "/*.*")
result_dir = os.path.join(args.result_dir, test_name)
checkdir(result_dir)
ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

print("\n CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        Img = cv2.imread(imgName, 1)
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv[:,:,0]/255.0
        start = time()
        Input_tensor = torch.from_numpy(Iorg_y)
        Input_tensor = Input_tensor.type(torch.FloatTensor)
        Input_tensor = Input_tensor.to(device).view(1, -1, Input_tensor.shape[0], Input_tensor.shape[1])
        out,_= model(Input_tensor)
        end = time()
        im_res_y = out.data[0].numpy().astype(np.float32)

        im_res_y = im_res_y * 255.
        im_res_y[im_res_y < 0] = 0
        im_res_y[im_res_y > 255.] = 255.
        im_res_y = im_res_y[0, :, :]
        # rec_PSNR = 0
        # rec_SSIM = 0
        rec_PSNR = psnr(im_res_y, Img_yuv[:,:,0].astype(np.float64))
        rec_SSIM = ssim(im_res_y, Img_yuv[:,:,0].astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = im_res_y  #只操作一个通道 其他类似
        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        resultName = imgName.replace(args.data_dir, args.result_dir)
        cv2.imwrite("%s_NLCSNet_ratio_%f_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(output_data)


print("CS Reconstruction End")

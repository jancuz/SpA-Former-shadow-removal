import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time

import torch
from torch.autograd import Variable

from utils import gpu_manage, heatmap
from SpA_Former import Generator

import glob
import os
from tqdm import tqdm

from torchvision import transforms
from PIL import Image

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cal_acc(prediction, label, thr = 128):
    prediction = (prediction > thr)
    print('prediction', prediction)
    label = (label > thr)
    print('label', label)
    prediction_tmp = prediction.astype(float)
    label_tmp = label.astype(float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union

def predict(args):

    gpu_manage(args)
    # Data Loading for ISTD dataset
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.root_path, 'test_A')) if f.endswith('.png')]
    data_path = [(os.path.join(args.root_path, 'test_A', img_name + '.png'), 
                os.path.join(args.root_path, 'test_B', img_name + '.png')) 
                for img_name in img_list]
    # Data Loading for ImageNet val.
    #img_list = [os.path.splitext(f)[0] for f in os.listdir(args.root_path) if f.endswith('.JPEG')]
    #data_path = [(os.path.join(args.root_path, img_name + '.JPEG'), 
    #            '*****') 
    #            for img_name in img_list]

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=args.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    print ('<=== Model loaded')
    
    TP, TN, Np, Nn = 0, 0, 0, 0
    ber_mean = 0

    # normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet
    # normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # SBU - same as ImageNet
    normal = transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
    # normal = transforms.Normalize([0.723, 0.616, 0.569], [0.169, 0.177, 0.197]) # ISIC2017
    img_transform = transforms.Compose([
        transforms.Resize((640,480)),
        transforms.ToTensor(),
        normal,
    ])

    for (img_path, target_path) in tqdm(data_path):
        print('===> Loading test image')
        img = cv2.imread(img_path, 1).astype(np.float32)

        #img = Image.open(img_path).convert('RGB')
        #print('img_img', np.asarray(img_img).shape)
        #img = img_transform(img).cuda()
        #print('img_transform', np.asarray(img_img).shape)

        # Convert the image to grayscale while keeping three channels
        print('===> Convert test image to grayscale')
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_g = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
        img = img / 255
        img = img.transpose(2, 0, 1)
        img = img[None]
        img_g = img_g / 255
        img_g = img_g.transpose(2, 0, 1)
        img_g = img_g[None]
        print ('<=== test image loaded')

        with torch.no_grad():
            x = torch.from_numpy(img)
            if args.cuda:
                x = x.cuda()

            print('===> Removing the shadow on an RGB image...')
            start_time = time.time()
            att, out = gen(x)
            print('<=== finish! %.3fs cost.' % (time.time()-start_time))

            x_ = x.cpu().numpy()[0]
            x_rgb = x_ * 255
            x_rgb = x_rgb.transpose(1, 2, 0).astype('uint8')
            out_ = out.cpu().numpy()[0]
            out_rgb = np.clip(out_[:3], 0, 1) * 255
            out_rgb = out_rgb.transpose(1, 2, 0).astype('uint8')
            att_ = att.cpu().numpy()[0] * 255
            att_heatmap = heatmap(att_.astype('uint8'))[0]
            att_heatmap = att_heatmap.transpose(1, 2, 0)

            x = torch.from_numpy(img_g)
            if args.cuda:
                x = x.cuda()
            print('===> Removing the shadow on an Grayscale image...')
            start_time = time.time()
            att_g, out_g = gen(x)
            print('<=== finish! %.3fs cost.' % (time.time()-start_time))
            x_ = x.cpu().numpy()[0]
            x_rgb_g = x_ * 255
            x_rgb_g = x_rgb_g.transpose(1, 2, 0).astype('uint8')
            out_ = out_g.cpu().numpy()[0]
            out_rgb_g = np.clip(out_[:3], 0, 1) * 255
            out_rgb_g = out_rgb_g.transpose(1, 2, 0).astype('uint8')
            att_ = att_g.cpu().numpy()[0] * 255
            att_heatmap_g = heatmap(att_.astype('uint8'))[0]
            att_heatmap_g = att_heatmap_g.transpose(1, 2, 0)

            allim = np.hstack((x_rgb, out_rgb, att_heatmap, x_rgb_g, out_rgb_g, att_heatmap_g))

            # save images
            filename = os.path.basename(img_path).split('.')
            filename1 = filename[0]
            filename2 = filename[1]
            if args.save_res_imgs:
                cv2.imwrite(args.save_filepath + filename1 + '_out_rgb.' + filename2, out_rgb)
                cv2.imwrite(args.save_filepath + filename1 + '_att_heatmap.' + filename2, att_heatmap)
                cv2.imwrite(args.save_filepath + filename1 + '_x_rgb_g.' + filename2, x_rgb_g)
                cv2.imwrite(args.save_filepath + filename1 + '_out_rgb_g.' + filename2, out_rgb_g)
                cv2.imwrite(args.save_filepath + filename1 + '_att_heatmap_g.' + filename2, att_heatmap_g)

            if args.GT_access:
                img_GT = cv2.imread(target_path)
                img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2GRAY)
                prediction = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2GRAY)
                TP_single, TN_single, Np_single, Nn_single, Union = cal_acc(prediction, img_GT)
                
                '''Calculate BER '''
                TP = TP + TP_single
                TN = TN + TN_single
                Np = Np + Np_single
                Nn = Nn + Nn_single
                ber_shadow = (1 - TP / Np) * 100
                ber_unshadow = (1 - TN / Nn) * 100
                ber_mean = 0.5 * (2 - TP / Np - TN / Nn) * 100
                print("Current ber is {}, shadow_ber is {}, unshadow ber is {}, Ntp/Np is {}".format(ber_mean, ber_shadow, ber_unshadow, TP/Np))
            
            # do shadow amout calculation and record it into the file
            if args.shadow_count:
                thr = 128
                N_all = prediction.shape[0] * prediction.shape[1]
                N_shadow = np.count_nonzero(prediction > thr)
                N_not_shadow = N_all - N_shadow
                shadow_a = N_shadow / N_all * 100

                with open(args.save_shadow_info_rgb, 'a') as f:
                    f.write(filename1+'_out_rgb.'+filename2+' '+str(N_shadow)+' '+str(N_not_shadow)+' '+str(N_all)+' '+str(shadow_a)+'\r\n')
    
    return ber_mean    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='./data/ISTD_Dataset/test', type=str)
    parser.add_argument('--shadow_count', default=False)
    parser.add_argument('--GT_access', default=True)
    #parser.add_argument('--test_filepath', default='./imgs/ImageNet/val_subset/*.JPEG', type=str)
    parser.add_argument('--pretrained', type=str, default='./results/000002-tra/models/gen_model_epoch_36.pth')
    parser.add_argument('--save_filepath', default='./imgs/ImageNet/val_subset/res_model33_epoch36/', type=str)
    parser.add_argument('--save_shadow_info_rgb', default='./data/ISTD_Dataset/test/res_model33_epoch36/image_shadow.txt', type=str)
    parser.add_argument('--save_res_imgs', default=False)
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=12)
    args = parser.parse_args()
    
    metric = predict(args)

    # write statistic (BER) for the model into the file 
    if args.GT_access:
        with open('./statistics/test_record.txt', 'a') as f:
            f.write(args.pretrained+' ')
            f.write(str(metric)+' '+args.root_path+'\r\n')
        print('Test ber results: {}'.format(metric))
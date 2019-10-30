import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import numpy as np
from read_data import sceneDisp
import torch.optim as optim

from gc_net import *
from python_pfm import *
import os, time
from guidedfilter import *

import re
import sys
from guidedfilter import *


model_path = './checkpoint/ckpt.t7'


def run2(left, right, max_points=100):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left,None)
    kp2, des2 = orb.detectAndCompute(right,None)
    #
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    #
    #maps = cv2.drawMatches(left,kp1,right,kp2,matches[:128],None, flags=2)
    # guide = GuidedFilter(left, 21, 0.00001)
    # m = cv2.medianBlur(left, 9)
    # g = guide.filter(m)
    # cv2.imwrite('g.png', (g*255).astype(np.uint8))
    #
    if len(matches) >= max_points:
        n_points = max_points
    else:
        n_points = len(matches / 2)
    #
    dist = []
    for i in range(n_points):
        qi = matches[i].queryIdx
        ti = matches[i].trainIdx
        d = kp1[qi].pt[0] - kp2[ti].pt[0] 
        if d != 0:
            dist.append(d)
    #
    dist = np.array(dist)
    if dist.mean() < 5:
        print('dist mean too small.')
        return 30
    else:
        return 0

#preprocess
def normalizeRGB(img):

    return img

tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

h=256
w=512
maxdisp=160 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=1
net = GcNet(h,w,maxdisp)

#input_L = sys.argv[1]#'../data/Synthetic/TL3.png'
#input_R = sys.argv[2]#'../data/Synthetic/TR3.png'
#output_path sys.argv[3]#= '../output/out.png'

#net=net.cuda()

net=torch.nn.DataParallel(net).cuda()

def to3D(arr):
    if len(arr.shape) < 3:
        return cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB) #* 1.5
        result = np.zeros((arr.shape[0], arr.shape[1], 3))
        result[:,:,0] = arr.copy()
        result[:,:,1] = arr.copy()
        result[:,:,2] = arr.copy()
    elif arr.shape[2] == 1:
        return cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB) #* 1.5
        result = np.zeros((arr.shape[0], arr.shape[1], 3))
        arr = np.reshape(arr, (arr.shape[0], arr.shape[1]))
        result[:,:,0] = arr.copy()
        result[:,:,1] = arr.copy()
        result[:,:,2] = arr.copy()
    else:
        return arr
    return result

def part(imL_in, imR_in, startH=0, startW=0):
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    imageL = imL_in[:, :, startH:(startH + h), startW:(startW + w)]
    imageR = imR_in[:, :, startH:(startH + h), startW:(startW + w)]
    #disL   = data['dispL'][:, :, startH:(startH + h), startW:(startW + w)]    
    imL.data.resize_(imageL.size()).copy_(imageL)
    imR.data.resize_(imageR.size()).copy_(imageR)
    #dispL.data.resize_(disL.size()).copy_(disL)
    loss_mul_list_test = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, h, w]) * d)).cuda()
        loss_mul_list_test.append(loss_mul_temp)
    loss_mul_test = torch.cat(loss_mul_list_test, 1)

    result=net(imL,imR)

    disp = torch.sum(result.mul(loss_mul_test),1)
    return disp

#test
def test(loadstate, input_L='../data/Real/TL5.bmp',
                    input_R='../data/Real/TR5.bmp',
                    output_path='../output/output.png'):
    if loadstate==True:
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accur']
    net.eval()

    start = time.time()
    
    left = cv2.imread(input_L)
    right = cv2.imread(input_R)

    shift = run2(left, right, max_points=100)
    out = test_oneshot(input_L, input_R, output_path, transform=tsfm,shift=shift)

    end = time.time() - start
    #print(end)
    return out 


def equalizeHist_color(img):

    #alpha = 1.5
    #beta = -30
    
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #cv2.imwrite('before_his.png', img ,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #cv2.imwrite('after_his.png', img_output,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #img_output = cv2.bilateralFilter(img_output, 15, 7,7)
    #img_output = img_output * alpha + beta
    ##img_output = cv2.medianBlur(img_output, 9)

    guide = GuidedFilter(img_output, 13, 0.00001)
    m = cv2.medianBlur(img_output, 7)
    img_output = guide.filter(m)
    
    return img_output

def test_oneshot(imL_name, imR_name, out_name, transform=None, shift=0):
    dispL = Variable(torch.FloatTensor(1).cuda())

    _L_in = cv2.imread(imL_name)#.transpose((2, 0, 1)).reshape(1,3,512,384)
    _R_in = cv2.imread(imR_name)#.transpose((2, 0, 1)).reshape(1,3,512,384)
    _L_in = _L_in[:,:_L_in.shape[1]-shift,:]
    _R_in = _R_in[:,shift:,:]
    _L_in = equalizeHist_color(_L_in)
    _R_in = equalizeHist_color(_R_in)

    tmp_h, tmp_w = _L_in.shape[0], _L_in.shape[1]
    ### method 1 resize ###
    _imageL = cv2.resize(_L_in, (w,h), cv2.INTER_LINEAR)
    _imageR = cv2.resize(_R_in, (w,h), cv2.INTER_LINEAR)
    _imageL, _imageR = to3D(_imageL), to3D(_imageR)
    
    data = {'imL': _imageL, 'imR': _imageR}
    if transform is not None:
        data['imL']=transform(data['imL']).reshape(1,3,h,w)
        data['imR']=transform(data['imR']).reshape(1,3,h,w)

    else:
        data['imL']=data['imL'].transpose((2, 0, 1)).reshape(1,3,h,w)
        data['imR']=data['imR'].transpose((2, 0, 1)).reshape(1,3,h,w)
    

    startH = 0#np.random.randint(0, 160)
    startW = 0#np.random.randint(0, 400)
 
    ### method resize ###
    with torch.no_grad():
        disp = part(data['imL'], data['imR'], 0, 0).data.cpu().numpy()


    im=disp.astype('uint8')#disp.data.cpu().numpy().astype('uint8')
    im=np.transpose(im,(1,2,0))
    ### method 1 ###
    im=im*tmp_w/w
    im = im.astype('uint8')
    im=cv2.resize(im, (tmp_w,tmp_h))
    ### method 2 ###
    #im = im[:,:384,:]
    #print('save file to', out_name, '......')
    #cv2.imwrite(out_name, im ,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return im


def main():
    load_state=True
    test(load_state)
    # for i in range(10):
    #     test(load_state, input_L='../data/Real/TL{}.bmp'.format(i),
    #                 input_R='../data/Real/TR{}.bmp'.format(i),
    #                 output_path='../output/real{}.png'.format(i))

if __name__=='__main__':
    main()
    # print('test accuracy less than 3 pixels:%f' %accuracy)
    #cv2.imwrite('test_err.png',diff,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def main():
    load_state=True
    test(load_state)
    # for i in range(10):
    #     test(load_state, input_L='../data/Real/TL{}.bmp'.format(i),
    #                 input_R='../data/Real/TR{}.bmp'.format(i),
    #                 output_path='../output/real{}.png'.format(i))

if __name__=='__main__':
    main()

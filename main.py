import numpy as np
import argparse
import cv2
import time
from core import *
from util import writePFM

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(args, Il, Ir):
    h, w, ch = Il.shape
    disp = np.zeros((h, w), dtype=np.int32)

    # TODO: Some magic
    disp = test(True, args.input_left, args.input_right)
    if disp.shape[1] != w:
        disp = cv2.GaussianBlur(disp,ksize=(5,5),sigmaX=1)
        disp = cv2.ximgproc.jointBilateralFilter(Il[:,:w-30,:],disp,d=10,sigmaColor=10.0,sigmaSpace=10.0)
        disp = cv2.resize(disp, (w,h))
    else:
        disp = cv2.GaussianBlur(disp,ksize=(5,5),sigmaX=1)
        disp = cv2.ximgproc.jointBilateralFilter(Il,disp,d=30,sigmaColor=10.0,sigmaSpace=10.0)
    return disp.astype(np.float32)


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(args, img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()

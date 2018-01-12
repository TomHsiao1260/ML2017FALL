import sys
import numpy as np
from skimage import io

INPUT = sys.argv[1]
RE_IMG = int(sys.argv[2].split('.')[0])
plot_name = 'reconstruction.jpg'
NUM_IMG = 415
SIZE_IMG = 600
SUM_EIGEN = 4

IMAGE = np.zeros((SIZE_IMG * SIZE_IMG * 3, NUM_IMG))
for i in range(NUM_IMG):
    img = io.imread('%s/%d.jpg' %(INPUT, i))
    img = img.flatten()
    IMAGE[:,i] = img
    
MEAN = np.mean(IMAGE, axis=1).reshape(-1,1)
U, s, V = np.linalg.svd(IMAGE-MEAN, full_matrices=False)

w = [0] * SUM_EIGEN
re_target = (IMAGE-MEAN)[:,RE_IMG]
re_img = np.zeros((SIZE_IMG * SIZE_IMG * 3, 1))
for i in range(SUM_EIGEN):
    w[i] = np.dot(U[:,i], re_target)
    re_img += (w[i] * U[:,i]).reshape(-1,1)
re_img += MEAN
re_img -= np.min(re_img)
re_img /= np.max(re_img)
re_img = (re_img * 255).astype(np.uint8)
re_img = re_img.reshape(SIZE_IMG, SIZE_IMG, 3)
io.imsave(plot_name, re_img)

    


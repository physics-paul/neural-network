"""*********************************************************

NAME:     Visualize Data

AUTHOR:   Paul Haddon Sanders IV, Ph.D

Date:     12/17/2018

*********************************************************"""

from scipy import io
from PIL import Image as im
import numpy as np

### Get data ###############################################

# There are 5000 training examples.
# Each training example is a 20x20 pixel image of the digit.
# Each pixel is represented by a number indicating
# greyscale intensity.
# This 20x20 grid of pixels is unrolled into a
# 400-dimensional vector.
# Then X is a 5000x400 matrix.
# Y represents the labels for each X training matrix.

mat = io.loadmat('ex4data1.mat')
X   = mat['X']
Y   = mat['y']
Y   = Y.reshape(Y.shape[0])
S   = X.shape[0]

### 1.1 Visualizing the data ###############################

# grab some random training data.

random_pos = np.random.choice(len(X),100,replace=False)

# now resize the array into a 10x10 grid of digits.

leng = (10,10,400)
image_arry = X[random_pos]
images = np.reshape(image_arry,leng)

# get the image ready.

all_im = im.new('L', (20 * leng[0],20 * leng[1]))

for i,row in enumerate(images):
    for j,ima in enumerate(row): 
    
        ima = np.reshape(ima,(20,20)).T
        ima = im.fromarray(ima * 255)

        all_im.paste(ima,(i * 20, j * 20))

all_im.show()

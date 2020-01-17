import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

plt.ion()

weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3) 
board = torch.tensor([[0,0,0,0,0], [0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0]])

b = board.view(1,1,5,5)

t = F.conv2d(b, weights, padding=1).view(5,5)
t = (t==12) | (t==3) | (t==13)
npt = np.int8(t) * 255
cv2.namedWindow("game", cv2.WINDOW_AUTOSIZE)

img = Image.fromarray(npt).convert('RGB')
img = np.array(img)

cv2.imshow("game", img)

#import ipdb; ipdb.set_trace()
cv2.waitKey(1000)
# import time
# time.sleep(2)

b = torch.tensor(t, dtype=torch.int64).view(1,1,5,5)

t = F.conv2d(b, weights, padding=1).view(5,5)
t = (t==12) | (t==3) | (t==13)
npt = np.int8(t) * 255
img = Image.fromarray(npt).convert('RGB')
img = np.array(img)
cv2.imshow("game", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
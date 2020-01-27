import cv2
import numpy as np
import sys
import torch
import torch.nn.functional as F

from PIL import Image

BOARD_HEIGHT = 200
BOARD_WIDTH = 300

distrib = torch.distributions.Bernoulli(0.5)

weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3) 
board = distrib.sample((BOARD_HEIGHT,BOARD_WIDTH)).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)
board = board.to(torch.int64)

cv2.namedWindow("game", cv2.WINDOW_NORMAL)

while True:
    newboard = F.conv2d(board, weights, padding=1).view(BOARD_HEIGHT,BOARD_WIDTH)
    newboard = (newboard==12) | (newboard==3) | (newboard==13)
    newboard_array = np.int8(newboard) * 255
    img = Image.fromarray(newboard_array).convert('RGB')
    img = np.array(img)
    cv2.imshow("game", img)
    q = cv2.waitKey(100)
    if q == 113: # 'q'
        cv2.destroyAllWindows()
        sys.exit(0)
    board = torch.tensor(newboard_array/255, dtype=torch.int64).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)

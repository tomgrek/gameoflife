import cv2
import numpy as np
import sys
import torch
import torch.nn.functional as F
from PIL import Image

BOARD_HEIGHT = 100
BOARD_WIDTH = 100

examples = torch.load('train.data')

# TODO support bigger batches, will have to change
# how the data is loaded.
trainloader = torch.utils.data.DataLoader(examples, batch_size=1,
                                          shuffle=False, num_workers=1)

cv2.namedWindow("game", cv2.WINDOW_NORMAL)

class GoL(torch.nn.Module):
    def __init__(self):
        super(GoL, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        self.fc1 = torch.nn.Linear(BOARD_HEIGHT*BOARD_WIDTH, BOARD_HEIGHT*BOARD_WIDTH)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x.view(-1))
        x = F.relu(x)
        return x.view(1,1,BOARD_HEIGHT,BOARD_WIDTH)

gol = GoL().cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(gol.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    last_frame = None
    for i, data in enumerate(trainloader):
        data = data.view(1,1,BOARD_HEIGHT,BOARD_WIDTH).to(torch.float32).to('cuda')
        if i == 0:    
            last_frame = data
            continue
        input_ = last_frame
        # Because each restart of the training datagen creates a totally new board
        # if the difference between the data and last_frame is > threshold, 
        # just skip this frame.
        target_ = data
        
        optimizer.zero_grad()

        output_ = gol(input_)
        loss = criterion(output_, target_) * 20 # loss on non-aliver is penalized more
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss /20))
            running_loss = 0.0
            newboard_array = np.int8(output_.cpu().clone().view(BOARD_HEIGHT,BOARD_WIDTH).detach()) * 255
            img = Image.fromarray(newboard_array).convert('RGB')
            img = np.array(img)
            cv2.imshow("game", img)
            q = cv2.waitKey(100)
        last_frame = data

print(gol.conv1.weight, output_, output_.mean(dim=-1))
#import ipdb; ipdb.set_trace()

distrib = torch.distributions.Bernoulli(0.5)
board = distrib.sample((BOARD_HEIGHT,BOARD_WIDTH)).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)
board = board.to(torch.float32).to('cuda')

# some problem with the below code
while True:
    newboard = gol(board)
    newboard_array = np.int8(newboard.cpu().clone().view(BOARD_HEIGHT,BOARD_WIDTH).detach()) * 255
    img = Image.fromarray(newboard_array).convert('RGB')
    img = np.array(img)
    cv2.imshow("game", img)
    q = cv2.waitKey(200)
    if q == 113: # 'q'
        cv2.destroyAllWindows()
        break
    if q == 114: # 'r'
        distrib = torch.distributions.Bernoulli(0.5)
        board = distrib.sample((BOARD_HEIGHT,BOARD_WIDTH)).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)
        board = board.to(torch.float32).to('cuda')
        newboard = board
        newboard_array = np.int8(newboard.cpu().clone().view(BOARD_HEIGHT,BOARD_WIDTH).detach()) * 255
        img = Image.fromarray(newboard_array).convert('RGB')
        img = np.array(img)
        cv2.imshow("game", img)
    board = newboard
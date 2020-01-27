import torch

data = torch.load('train.data')

last_frame = None
newdata = []

for i, d in enumerate(data):
    if i == 0:
        last_frame = d
        continue
    if abs(d.sum().item()) > 1.5 * abs(last_frame.sum().item()):
        pass
    else:
        newdata.append((last_frame, d))
    last_frame = d

torch.save(newdata, 'train.data.batch')
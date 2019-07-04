import torch
from utils.datasets import BDD100k
dataloader = torch.utils.data.DataLoader(
    BDD100k("/dataset/bdd1/label/traindrive_train.txt"), batch_size=32, shuffle=False, num_workers=2
)
for batch_i, (tt, imgs, targets) in enumerate(dataloader):
    if batch_i>0:
        break
    print(targets)
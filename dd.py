import os
import torch
import argparse
import numpy as np
import time


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        return x + x


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
print(args.local_rank)
torch.cuda.set_device(args.local_rank)
model = TestModel().cuda()
a = torch.from_numpy(np.array([1])).to("cuda:{}".format(args.local_rank))
torch.distributed.init_process_group(backend="nccl", init_method='env://')
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
print("success {}".format(args.local_rank))
# print(model.cuda())
c_sec = time.time()
epoch = 1000
for _ in range(epoch):
    torch.distributed.reduce(a, dst=0)

if not args.local_rank:
    print(a.cpu().numpy()[0], (time.time()-c_sec)/epoch)
np.maximum()
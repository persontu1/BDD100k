from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="/dataset/PyTorch-YOLOv3/data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="/dataset/PyTorch-YOLOv3/config/bdd100k.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="/dataset/PyTorch-YOLOv3/config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="/dataset/PyTorch-YOLOv3/weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="/dataset/PyTorch-YOLOv3/data/bdd100k.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--nums_gpu", type=int, default=1, help="number of gpu")
parser.add_argument("--local_rank", type=int, default=1, help="current rank")
parser.add_argument("--train_path", type=str, default="/dataset/bdd1/label/traindrive_train.txt", help="train info")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda
torch.cuda.set_device(opt.local_rank)
os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = opt.train_path

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"]) * opt.nums_gpu
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
# model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

dataset = BDD100k(train_path)
model1 = model
device = None
train_sampler = None
if opt.nums_gpu>1:
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    print("success {}".format(opt.local_rank))
    device = torch.device("cuda:{}".format(opt.local_rank))
    print("device {} is set".format("cuda:{}".format(opt.local_rank)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
# Get dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, sampler=train_sampler
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    for batch_i, (tt, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        if opt.nums_gpu > 1:
            imgs = imgs.to(device)
        # temp = targets.numpy()[0]
        targets = Variable(targets.type(Tensor), requires_grad=False)
        if opt.nums_gpu > 1:
            targets = targets.to(device)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        if not opt.local_rank:
            # torch.distributed.reduce(loss, 0)
            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    opt.epochs,
                    batch_i,
                    len(dataloader),
                    model1.losses["x"],
                    model1.losses["y"],
                    model1.losses["w"],
                    model1.losses["h"],
                    model1.losses["conf"],
                    model1.losses["cls"],
                    loss.item(),
                    model1.losses["recall"],
                    model1.losses["precision"],
                )
            )

        model1.seen += imgs.size(0)

    if not opt.local_rank and epoch % opt.checkpoint_interval == 0:
        print("save model")
        model1.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))

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
from utils.uitls1 import *
from torch.nn.functional import upsample
from torch.optim import Adam

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
concatLayer = ConcatLayer()
preprocessLayer = PreprocessLayerV1()
# model.apply(weights_init_normal)
img_path = "data/coco/images/train2014/COCO_train2014_000000291797.jpg"
label_path = "data/coco/labels/train2014/COCO_train2014_000000291797.txt"
tag_id = "train2014_000000291797"
target_label = 22
original_label_pos = 1

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

input_img, filled_labels = preprocess_image(img_path,
                                            label_path)
original_label = filled_labels[original_label_pos, 0]
one_object_per_image(filled_labels, original_label)
input_img = input_img.unsqueeze(0)/255
filled_labels = filled_labels.unsqueeze(0)
for k, v in model.named_parameters():
    v.requires_grad = False
    print(v.requires_grad)

img = Variable(input_img.type(Tensor))
target = Variable(filled_labels.type(Tensor), requires_grad=False)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
with torch.no_grad():
    detection = model(img)
    detection = non_max_suppression(detection, 80, 0.5, 0.4)
    temp = detection[0].cpu().numpy()
    detection = detection[0]
# x1, y1, x2, y2 = temp[0, 0], temp[0, 1], temp[0, 2], temp[0, 3]
detect_result1 = objects_and_confidences(detection)
return_coordinates_and_draw_boxes(detection, np.array(Image.open(img_path)),'normal2.png')
x1, y1, x2, y2, class_pred = return_coordinates_v2(detection, img_path, original_label)
img = Variable(torch.Tensor(preprocess_image_v2(img_path)).type(Tensor)).unsqueeze(0)/255
# img = Variable(img.type(Tensor))
target_img = img[:, int(y1):int(y2), int(x1):int(x2), :]
top = img[:, int(y2):, :, :]
bottom = img[:, :int(y1), :, :]
left = img[:, int(y1):int(y2), :int(x1), :]
right = img[:, int(y1):int(y2), int(x2):, :]

ad_img = target_img.detach().clone()
ad_img.requires_grad = True
# middle = torch.cat((left, ad_img, right), 2)
# ad_input_img = torch.cat((bottom, middle, top), 3)
ad_labels = filled_labels.clone()
ad_labels.data[0, original_label_pos, 0] = target_label
optimizer = Adam([ad_img])
MSELoss = torch.nn.MSELoss(size_average=True)
L1Loss = torch.nn.L1Loss(size_average=True)
#
# test_img = target_img.detach().clone()
# test_img[...] = 128
# Image.fromarray(concatLayer(test_img, top, bottom, left, right).cpu().squeeze().numpy().astype(np.uint8))\
#     .save("aaa.png", "PNG")

steps = 3000
# print((ad_input_img.cpu().detach().numpy()[..., int(x1):int(x2), int(y1):int(y2)] == ad_img.cpu().detach().numpy()).all())

top1 = top.clone()
for i in range(steps):
    optimizer.zero_grad()
    ad_labels = Variable(ad_labels.type(Tensor), requires_grad=False)
    base_loss = model(preprocessLayer(concatLayer(torch.clamp(ad_img, 0, 1), top, bottom, left, right)), ad_labels)
    mse_loss = MSELoss(ad_img, target_img)
    l1_loss = L1Loss(ad_img, target_img)
    loss = base_loss + 0.1 * l1_loss + mse_loss

    loss.backward()
    optimizer.step()
    save_grad_pic(ad_img,'grad',i)
    # index = return_topk_grad(torch.abs(ad_img.grad))
    # xxx = ad_img[index[0], index[1], index[2], index[3]] \
    #                                                 - torch.sign(ad_img)[index[0], index[1], index[2], index[3]]

    detection = model(preprocessLayer(concatLayer(torch.clamp(ad_img, 0, 1), top, bottom, left, right)))
    detection = non_max_suppression(detection, 80, 0.5, 0.4)
    temp = detection[0].cpu().numpy()
    detection = detection[0]
    detect_result1 = objects_and_confidences(detection)
    if target_label in detect_result1:
        print('target reached')
        break
    # ad_result = concatLayer(torch.clamp((target_img - ad_img.grad.sign()).cpu().detach(),0,255), top.cpu().detach(), bottom.cpu().detach(),
    #                         left.cpu().detach(), right.cpu().detach())
    # ppp = Image.fromarray(ad_result.numpy().astype(np.uint8))
    # ppp.save("new_ad1.png", "PNG")
    # return_coordinates_and_draw_boxes(detection, ad_result.numpy().astype(np.uint8),'ad2.png')
    # _, _, _, _, class_pred1 = return_coordinates(detection, img_path)

    # if class_pred != class_pred1:
    #     break



    print("loss: {}  base_loss: {}  mse_loss: {}  conf: {}  cls_loss: {}  recall: {}  precision: {}".format(
                loss.item(),
                base_loss.item(),
                mse_loss.item(),
                model.losses["conf"],
                model.losses["cls"],
                model.losses["recall"],
                model.losses["precision"]))

print()
with torch.no_grad():
    ad_result = concatLayer(torch.clamp(ad_img, 0, 1), top, bottom, left, right)
    detection = model(preprocessLayer(ad_result))
    detection = non_max_suppression(detection, 80, 0.5, 0.4)
    temp = detection[0].cpu().numpy()
    detection = detection[0]
    # x1, y1, x2, y2,class_pred = return_coordinates(detection, img_path)
    # ad_result = concatLayer(torch.clamp(torch.round(ad_img.cpu().detach()), 0, 255), top.cpu().detach(), bottom.cpu().detach(),
    #                         left.cpu().detach(), right.cpu().detach())
    final_result = torch.round(ad_result*255)
    ppp = Image.fromarray(final_result.cpu().squeeze().numpy().astype(np.uint8))
    return_coordinates_and_draw_boxes(detection, final_result.cpu().squeeze().numpy().astype(np.uint8), 'ad2.png')
    ppp.save("{}.png".format(tag_id), "PNG")
# for epoch in range(opt.epochs):
#     for batch_i, (_, imgs, targets) in enumerate(dataloader):
#         imgs = Variable(imgs.type(Tensor))
#         temp=targets.numpy()[0]
#         targets = Variable(targets.type(Tensor), requires_grad=False)  # trainable
#
#         optimizer.zero_grad()
#
#         loss = model(imgs, targets)
#
#         loss.backward()
#         optimizer.step()
#
#         print(
#             "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
#             % (
#                 epoch,
#                 opt.epochs,
#                 batch_i,
#                 len(dataloader),
#                 model.losses["x"],
#                 model.losses["y"],
#                 model.losses["w"],
#                 model.losses["h"],
#                 model.losses["conf"],
#                 model.losses["cls"],
#                 loss.item(),
#                 model.losses["recall"],
#                 model.losses["precision"],
#             )
#         )
#
#         model.seen += imgs.size(0)
#         break
#
#     if epoch % opt.checkpoint_interval == 0:
#         model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))

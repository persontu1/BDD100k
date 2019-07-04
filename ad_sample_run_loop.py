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
import pickle


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
os.makedirs("ad_output", exist_ok=True)
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

count_map = []
pros_results = []
now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
os.makedirs("output/{}".format(now_time), exist_ok=True)
is_double = True

def run_main(target_label):
    torch.cuda.empty_cache()
    # Initiate model
    Tensor = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor
    Tensor = Tensor if is_double else torch.cuda.FloatTensor if cuda else torch.FloatTensor
    model = Darknet(opt.model_config_path).type(Tensor)
    model.load_weights(opt.weights_path)
    concatLayer = ConcatLayer()
    preprocessLayer = PreprocessLayer()
    # model.apply(weights_init_normal)
    img_path = "data/coco/images/train2014/COCO_train2014_000000291797.jpg"
    label_path = "data/coco/labels/train2014/COCO_train2014_000000291797.txt"
    original_label_pos = 1
    # tag_id = "train2014_000000291797"
    # target_label = 22

    if cuda:
        model = model.cuda()

    model.train()


    input_img, filled_labels = preprocess_image(img_path,
                                                label_path)
    original_label = filled_labels[original_label_pos, 0]
    one_object_per_image(filled_labels, original_label)
    input_img = input_img.unsqueeze(0)
    filled_labels = filled_labels.unsqueeze(0)
    for k, v in model.named_parameters():
        v.requires_grad = False
        print(v.requires_grad)

    img = Variable(input_img.type(Tensor))
    target = Variable(filled_labels.type(Tensor), requires_grad=False)
    with torch.no_grad():
        detection = model(img)
        detection = non_max_suppression(detection, 80, 0.8, 0.4, is_double=is_double)
        detection = detection[0]

    detect_result1 = objects_and_confidences(detection)
    return_coordinates_and_draw_boxes(detection, np.array(Image.open(img_path)), 'normal2.png')
    x1, y1, x2, y2, class_pred = return_coordinates_v2(detection, img_path, original_label)
    img = Variable(torch.Tensor(preprocess_image_v2(img_path)).type(Tensor)).unsqueeze(0)
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
    optimizer = PixelOtimizer([ad_img], lr=1)
    MSELoss = torch.nn.MSELoss(size_average=True)
    L1Loss = torch.nn.L1Loss(size_average=True)

    test_img = target_img.detach().clone()
    test_img[...] = 128
    Image.fromarray(concatLayer(test_img, top, bottom, left, right).cpu().squeeze().numpy().astype(np.uint8)) \
        .save("aaa.png", "PNG")

    steps = 100
    pre_half_count_map = {}
    post_half_count_map = {}
    for i in range(steps):
        optimizer.zero_grad()
        ad_labels = Variable(ad_labels.type(Tensor), requires_grad=False)
        base_loss = model(preprocessLayer(concatLayer(torch.clamp(ad_img, 0, 255), top, bottom, left, right)),
                          ad_labels)
        mse_loss = MSELoss(ad_img, target_img)
        l1_loss = L1Loss(ad_img, target_img)
        loss = base_loss + l1_loss + mse_loss

        loss.backward()
        optimizer.step()

        detection = model(preprocessLayer(concatLayer(ad_img, top, bottom, left, right)))
        detection = non_max_suppression(detection, 80, 0.8, 0.4, is_double=is_double)
        temp = detection[0].cpu().numpy()
        detection = detection[0]
        detect_result1 = objects_and_confidences(detection)
        if i < steps / 2:
            for re in detect_result1:
                pre_half_count_map[classes[re]] = pre_half_count_map.setdefault(classes[re], 0) + 1
        else:
            for re in detect_result1:
                post_half_count_map[classes[re]] = post_half_count_map.setdefault(classes[re], 0) + 1

        # if target_label in detect_result1:
        #     print('target reached')
        #     break

        print("step: {}th  loss: {}  base_loss: {}  mse_loss: {}  conf: {}  cls_loss: {}  recall: {}  precision: {}".format(
            i,
            loss.item(),
            base_loss.item(),
            mse_loss.item(),
            model.losses["conf"],
            model.losses["cls"],
            model.losses["recall"],
            model.losses["precision"]))

    print()
    with torch.no_grad():
        ad_result = concatLayer(torch.clamp(torch.round(ad_img), 0, 255), top, bottom, left, right)
        detection = model(preprocessLayer(ad_result))
        detection = non_max_suppression_v1(detection, 80, 0.8, 0.4, is_double=is_double)
        class_pros = detection[0][:, 7:]
        pros_results.append(class_pros)
        detection = detection[0][:, 0:7]

        ppp = Image.fromarray(ad_result.cpu().squeeze().numpy().astype(np.uint8))
        return_coordinates_and_draw_boxes(detection, ad_result.cpu().squeeze().numpy().astype(np.uint8),
                                          'output/{}/ad_{}.png'.format(now_time, target_label))
        ppp.save("ad_output/{}.png".format(target_label), "PNG")
    count_map.append(dict(pre_half_count_map=pre_half_count_map, post_half_count_map=post_half_count_map))


if __name__ == '__main__':
    # for j in range(80):
    #     run_main(j)
    # pros_r = [i.cpu().numpy() for i in pros_results]
    # pickle.dump(pros_r, open('pros_results.pkl', 'wb'))
    # torch.nn.KLDivLoss
    run_main(5)
    # run_main(5)
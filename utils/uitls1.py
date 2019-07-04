import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
from utils.utils import *
from torch.nn.functional import upsample, pad
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from functools import reduce


def preprocess_image(img_path="/root/PyTorch-YOLOv3/data/coco/images/train2014/COCO_train2014_000000291797.jpg",
                     label_path="/root/PyTorch-YOLOv3/data/coco/labels/train2014/COCO_train2014_000000291797.txt"):
    img = np.array(Image.open(img_path))

    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize
    input_img1 = torch.Tensor(input_img).unsqueeze(0).permute(0,3,1,2)
    input_img1 = upsample(input_img1,size=(416,416),mode='bilinear').squeeze(0)
    input_img = resize(input_img, (416, 416, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()
    print((input_img1.numpy()==input_img.numpy()).all())

    # ---------
    #  Label
    # ---------
    labels = None
    if os.path.exists(label_path):
        labels = np.loadtxt(label_path).reshape(-1, 5)
        # Extract coordinates for unpadded + unscaled image
        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
    # Fill matrix
    filled_labels = np.zeros((50, 5))
    if labels is not None:
        filled_labels[range(len(labels))[:50]] = labels[:50]
    filled_labels = torch.from_numpy(filled_labels)
    return input_img1, filled_labels


def preprocess_image_v2(img_path="/root/PyTorch-YOLOv3/data/coco/images/train2014/COCO_train2014_000000291797.jpg"):
    return np.array(Image.open(img_path))


def return_coordinates(detections, path="data/coco/images/train2014/COCO_train2014_000000291797.jpg"):
    classes = load_classes("data/coco.names")
    img = np.array(Image.open(path))
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            if i == 0:
                continue
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            # box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            # box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            # y2 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            # x2 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            return x1, y1, x2, y2, classes[int(cls_pred)]


def return_coordinates_v2(detections, path="data/coco/images/train2014/COCO_train2014_000000291797.jpg", original_label=0):
    classes = load_classes("data/coco.names")
    img = np.array(Image.open(path))
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            if int(cls_pred) != original_label:
                continue
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            y2 = y1 + box_h
            x2 = x1 + box_w
            return x1 if x1 > 0 else 0, \
                   y1 if y1 > 0 else 0, \
                   x2 if x2 < img.shape[1] else img.shape[1], \
                   y2 if y2 < img.shape[0] else img.shape[0], \
                   classes[int(cls_pred)]


cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
classes = load_classes('data/coco.names')


def return_coordinates_and_draw_boxes(detections, img, img_name):
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def objects_and_confidences(detections):
    result=[]
    if detections is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            result.append(int(cls_pred))
    return np.array(result)


def return_topk_grad(data, grad, abs_grad, k=1):
    data_flatten = data.view(reduce(lambda x, y: x*y, data.shape))
    sign_flatten = torch.sign(grad).view(reduce(lambda x, y: x*y, grad.shape))
    flatten = abs_grad.view(reduce(lambda x, y: x*y, abs_grad.shape))
    topk = flatten.topk(k)[1]
    return data_flatten, sign_flatten, topk


def one_object_per_image(filled_labels, res_pos):
    for i, row in enumerate(filled_labels):
        if row[0] != res_pos:
            for j, col in enumerate(row):
                filled_labels.data[i, j] = 0


def non_max_suppression_v1(prediction, num_classes, conf_thres=0.5, nms_thres=0.4, is_double=False):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.double(), class_pred.double(), image_pred[:, 5 : 5 + num_classes]), 1) if is_double\
            else torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float(), image_pred[:, 5 : 5 + num_classes]), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, 6].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, 6] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def save_grad_pic(ad_img, grad_dir='', steps=0):
    os.makedirs(grad_dir, exist_ok=True)
    p = Image.fromarray(torch.round((torch.abs(ad_img.grad).squeeze() - torch.abs(ad_img.grad).squeeze().min()) / (
                torch.abs(ad_img.grad).squeeze().max() - torch.abs(
            ad_img.grad).squeeze().min()) * 255).cpu().numpy().astype(np.uint8))
    p.save('{}/grad_{}.png'.format(grad_dir, steps), 'PNG')
    grad1 = ad_img.grad.cpu().clone()
    grad1[grad1 < 0] = 0
    min_max_norm = (grad1 - grad1.min()) / (grad1.max() - grad1.min())
    grad1_pic = torch.round(min_max_norm * 255)
    p1 = Image.fromarray(grad1_pic.numpy().astype(np.uint8).squeeze())
    p1.save('{}/grad_gt0_{}.png'.format(grad_dir, steps), 'PNG')



class PixelOtimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1, k=1, full=True):
        super(PixelOtimizer, self).__init__(params, dict(lr=lr, k=k, full=full))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                sign = torch.sign(grad)
                abs_grad = torch.abs(grad)
                if not group['full']:
                    data_flatten, sign_flatten, index = return_topk_grad(p, grad, abs_grad, k=group['k'])
                    data_flatten.data[index] -= sign_flatten[index]
                else:
                    p.data -= group['lr'] * sign
        return loss

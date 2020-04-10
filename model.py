import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchsummary import summary

from PIL import Image
from PIL import ImageFile

import numpy as np
import io
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_instance_segmentation_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transforms():
    return T.Compose([T.ToTensor()])

def transform_image(img_bytes):
    img_transforms = get_transforms()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img_transforms(img).unsqueeze(0)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr>=0.5
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

model_path = os.path.join("models", "sky-region_mask_r-cnn_resnet50-fpn-1579167716")
model = get_instance_segmentation_model()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def get_mask_image(img_bytes):
    img = transform_image(img_bytes)

    with torch.no_grad():
        prediction = model(img)

    mask = 1 - prediction[0]['masks'][0, 0] # Creating Negative Mask
    mask = Image.fromarray(mask.mul(255).byte().numpy())
    return mask

def get_maoe(img_bytes):
    img = transform_image(img_bytes)

    with torch.no_grad():
        prediction = model(img)

    mask = prediction[0]['masks'][0, 0].numpy()
    maoe = first_nonzero(mask, axis=0, invalid_val=mask.shape[0]-1).tolist()
    return maoe
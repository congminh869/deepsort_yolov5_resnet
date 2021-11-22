import time
import os
import scipy.io
import yaml
import math
import timm
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, models, transforms

from PIL import Image




#model 



######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
#predict



class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.name = 'ft_ResNet50'
        self.which_epoch ='last'
        self.gpu_ids = '0'
        self.batchsize = 1
        # test_dir = '/home/minh/Documents/minh/work/object_tracking_yolov5/Yolov5_DeepSort_Pytorch/Person_reID_baseline_pytorch/Market/pytorch'
        # load the training config
        self.config_path = '/home/minh/Documents/minh/work/object_tracking_yolov5/Yolov5_DeepSort_Pytorch/Person_reID_baseline_pytorch/model/ft_ResNet50/opts.yaml'
        with open(self.config_path, 'r') as stream:
                config = yaml.safe_load(stream)
        self.fp16 = config['fp16'] 
        self.PCB = config['PCB']
        self.use_dense = config['use_dense']
        self.use_NAS = config['use_NAS']
        self.use_swin = config['use_swin']
        self.stride = config['stride']
        self.nclasses = config['nclasses']
        self.ibn = config['ibn']
        self.data_transforms = transforms.Compose([
            transforms.Resize((64, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        mss = '1'
        str_ms = mss.split(',')
        self.ms = []
        for s in str_ms:
            s_f = float(s)
            self.ms.append(math.sqrt(s_f))

        #load model 
        self.model = ft_net(self.nclasses, stride = self.stride, ibn = self.ibn )
        save_path_checkpoint = '/home/minh/Documents/minh/work/object_tracking_yolov5/Yolov5_DeepSort_Pytorch/Person_reID_baseline_pytorch/model/ft_ResNet50/net_%s.pth'%self.which_epoch
        self.model.load_state_dict(torch.load(save_path_checkpoint))
        self.model.classifier.classifier = nn.Sequential()
        self.model = self.model.eval()
        self.model = self.model.cuda()


    def fliplr(self,img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def load_image(self, path):
        # path = path[0]
        src = Image.fromarray(path)
        src = self.data_transforms(src)
        src = src.unsqueeze(dim=0)
        return src


    def __call__(self, im_crops):
        features =torch.FloatTensor().cuda()
        # image_path = im_crops
        for image_path in im_crops:
            src = self.load_image(image_path)
            # print('===============================',image_path.shape)
            img = src
            # print(img.shape)
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n,512).zero_().cuda()
            for i in range(2):
                if(i==1):
                    img = self.fliplr(img)
                input_img = Variable(img.cuda())
                for scale in self.ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = self.model(input_img) 
                    ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff.data), 0)
        # print('ff-feature : ', ff.data.cpu().numpy().shape) #torch.Size([1, 512])
        return features.cpu().numpy() #feature.shape :  (1, 512)


if __name__ == '__main__':
    img = '/home/minh/Documents/minh/work/object_tracking_yolov5/Yolov5_DeepSort_Pytorch/Person_reID_baseline_pytorch/Market/pytorch/demo_minh/0000/0000_c1s1_000151_01.jpg'
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

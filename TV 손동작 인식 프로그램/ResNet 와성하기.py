import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet as resnet
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from torchvision import models
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

import torch
print("PyTorch version: {}".format(torch.__version__))
print("CUDA version: {}".format(torch.version.cuda))
print(torch.cuda.get_device_name(0))



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device =torch.devi ce('cpu')
CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':30,# 10
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,#4
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('./train.csv')

train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list # 데이터 전처리

    def __getitem__(self, index): # 데이터셋에서 1개의 샘플을 가져오는 함수
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self): # 데이터셋의 길이. 총샘플수 가져오기
        return len(self.video_path_list)

    # 영상 데이터 전처리


    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            ret, img = cap.read()
            base_img = cv2.imread('base_img.png')

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(base_img,hand_landmarks, mp_hands.HAND_CONNECTIONS)
            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.

            frames.append(img)

        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

# transform,
class CustomDataset(Dataset):

    def __init__(self, video_path_list, label_list, transform=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            ret, img = cap.read()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 3))
            img[:, :, 0] = clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.

            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']

            frames.append(img)

        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

train_transform = A.Compose([
    # A.SmallestMaxSize(max_size=160),
    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1),
    # A.RandomCrop(height=128, width=128),
    # A.CLAHE(p=1),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    # A.RGBShift(r_shift_limit=2, g_shift_limit=2, b_shift_limit=2, p=1),
    # A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0, p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),max_pixel_value=1.0,p=1),
    # A.pytorch.transforms.ToTensorV2()
    ])

# 데이터값 추출
# file_paths, labels, transform=None
train_dataset = CustomDataset(train['path'].values, train['label'].values,train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# 데이터값 추출
val_dataset = CustomDataset(val['path'].values, val['label'].values,train_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
#
# model = models.resnet50(pretrained=True).to(device)
#===============
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200', 'get_fine_tuning_parameters'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 spatial_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(spatial_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

##########################################################################################
##########################################################################################


def get_fine_tuning_parameters(model, ft_prefixes):

    assert isinstance(ft_prefixes, str)

    if ft_prefixes == '':
        print('WARNING: training full network because --ft_predixes=None')
        return model.parameters()

    print('#'*60)
    print('Setting finetuning layer prefixes: {}'.format(ft_prefixes))

    ft_prefixes = ft_prefixes.split(',')
    parameters = []
    param_names = []
    for param_name, param in model.named_parameters():
        for prefix in ft_prefixes:
            if prefix in param_name:
                print('  Finetuning parameter: {}'.format(param_name))
                parameters.append({'params': param, 'name': param_name})
                param_names.append(param_name)

    for param_name, param in model.named_parameters():
        if param_name not in param_names:
            # This sames a lot of GPU memory...
            print('disabling gradient for: {}'.format(param_name))
            param.requires_grad = False

    return parameters



# def get_fine_tuning_parameters(model, ft_begin_index):
#
#     assert isinstance(ft_begin_index, int)
#     if ft_begin_index == 0:
#         print('WARNING: training full network because --finetune_begin_index=0')
#         return model.parameters()
#
#     for param_name, param in model.named_modules():
#         print(param_name)
#
#
#     ft_module_names = []
#     for i in range(ft_begin_index, 5):
#         ft_module_names.append('layer{}'.format(i))
#     ft_module_names.append('fc')
#
#     print('Modules to finetune: {}'.format(ft_module_names))
#
#     parameters = []
#     param_names_to_finetune = []
#     for k, v in model.named_parameters():
#         for ft_module in ft_module_names:
#             if ft_module in k:
#                 parameters.append({'params': v, 'name': k})
#                 param_names_to_finetune.append(k)
#                 break
#         else:
#             parameters.append({'params': v, 'lr': 0.0, 'name': k})
#             param_names_to_finetune.append(k)
#
#     # Disabling gradients for frozen weights (hacky...)
#     frozen_module_names = []
#     for i in range(0, ft_begin_index):
#         frozen_module_names.append('layer{}'.format(i))
#     for k, v in model.named_parameters():
#         for frozen_module in frozen_module_names:
#             if frozen_module in k:
#                 print('disabling grad for: {}'.format(k))
#                 v.requires_grad = False
#     model.module.conv1.requires_grad = False
#     model.module.bn1.requires_grad = False
#
#     return parameters

##########################################################################################
##########################################################################################

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
#===============
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

model_depth = 34

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ==========================
# class BaseModel(nn.Module):
#     def __init__(self, num_classes=5):
#         super(BaseModel, self).__init__()
#         self.feature_extract = nn.Sequential(
#             nn.Conv3d(3, 8, (3, 3, 3)), # 3 - 8
#             nn.ReLU(), # 활성화함수
#             nn.BatchNorm3d(8), #정규화
#             nn.MaxPool3d(2), # 풀링
#             nn.Conv3d(8, 32, (2, 2, 2)), # 8 - 32
#             nn.ReLU(),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, (2, 2, 2)), # 32- 64
#             nn.ReLU(),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, (2, 2, 2)), # 64 - 128
#             nn.ReLU(),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d((1, 7, 7)),
#         )
#         self.classifier = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.feature_extract(x)
#         x = x.view(batch_size, -1)
#         x = self.classifier(x)
#         return x



# ===============================
# print((q-w+(2*r))/e+1)
# class ResBlock(nn.Module):
# 	def __init__(self, block):
# 		super().__init__()
# 		self.block = block
# 	def forward(self, x):
# 		return self.block(x) + x #f(x) + x
#
# class Conv6Res(nn.Module):
#   def __init__(self, num_classes=5):
#     super().__init__()
#     self.name = 'conv6res'
#     self.model = nn.Sequential(
#         nn.Conv3d(3, 64, 7, 2, 4),
#         nn.ReLU(),
#         nn.BatchNorm3d(64),
#         nn.MaxPool3d(2),
#         nn.Conv3d(8, 32, 2, 1, 1),
#         nn.ReLU(),
#         nn.BatchNorm3d(32),
#         nn.MaxPool3d(2),
#         nn.Conv3d(32, 64, 2, 1, 1), # 32- 64
#         nn.ReLU(),
#         nn.BatchNorm3d(64),
#         nn.MaxPool3d(2),
#         nn.Conv3d(64, 128, 2, 1, 1), # 64 - 128
#         nn.ReLU(),
#         nn.BatchNorm3d(128),
#         nn.MaxPool3d(2),
#
#         ResBlock(
#             nn.Sequential(
#                 nn.Conv3d(128, 128, 2),
#                 nn.ReLU(),
#                 # nn.BatchNorm3d(128),
#                 nn.MaxPool3d(2),
#
#                 nn.Conv3d(128, 128, 2),
#                 nn.ReLU(),
#                 # nn.BatchNorm3d(128),
#                 nn.MaxPool3d(2))),
#
#         ResBlock(
#             nn.Sequential(
#                 nn.Conv3d(128, 128, 2),
#                 nn.ReLU(),
#                 # nn.BatchNorm3d(128),
#                 nn.MaxPool3d(2),
#
#                 nn.Conv3d(128, 128, 2),
#                 nn.ReLU(),
#                 # nn.BatchNorm3d(128),
#                 nn.MaxPool3d(2),
#
#                 nn.Conv3d(128, 128, 2,1,1),
#                 nn.ReLU(),
#                 # nn.BatchNorm3d(128),
#                 nn.MaxPool3d(2))),
#         nn.MaxPool3d((1, 7, 7)),
#     )
#     self.classifier = nn.Linear(512, num_classes)
#
#   def forward(self, x):
#       batch_size = x.size(0)
#       x = self.model(x)
#       x = x.view(batch_size, -1)
#       x = self.classifier(x)
#       return x
# ==========================
# q,w,e,r = 128,7,2,4
# print(((q-w+(2*r))/e)+1)
#
# def conv_start():
#     return nn.Sequential(
#         nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=4),
#         nn.BatchNorm3d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool3d(kernel_size=3, stride=2),
#     )
# def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
#     layers = []
#     if down:
#         layers.append(nn.Conv3d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
#     else:
#         layers.append(nn.Conv3d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
#     layers.extend([
#         nn.BatchNorm3d(mid_dim),
#         nn.ReLU(inplace=True),
#         nn.Conv3d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(mid_dim),
#         nn.ReLU(inplace=True),
#         nn.Conv3d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
#         nn.BatchNorm3d(out_dim),
#     ])
#     return nn.Sequential(*layers)
#
# class Bottleneck(nn.Module):
#     def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
#         super(Bottleneck, self).__init__()
#         if starting:
#             down = False
#         self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
#         self.relu = nn.ReLU(inplace=True)
#         if down:
#             conn_layer = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=2, padding=0), # size 줄어듬
#         else:
#             conn_layer = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0), # size 줄어들지 않음
#
#         self.changedim = nn.Sequential(conn_layer, nn.BatchNorm3d(out_dim))
#
#     def forward(self, x):
#         identity = self.changedim(x)
#         x = self.block(x)
#         x += identity
#         x = self.relu(x)
#         return x
#
# def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
#     layers = []
#     layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
#     for _ in range(1, repeats):
#         layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
#     return nn.Sequential(*layers)
#
#
# class ResNet(nn.Module):
#     def __init__(self, repeats: list = [3, 4, 6, 3], num_classes=5):
#         # resnet = ResNet(repeats=[3,4,23,3], num_classes = 10) # 101 Layer
#         # resnet = ResNet(repeats=[3,8,36,3], num_classes = 10) # 152 Layer
#         super(ResNet, self).__init__()
#         self.num_classes = num_classes
#         # 1번
#         self.conv1 = conv_start()
#
#         # 2번
#         base_dim = 64
#         self.conv2 = make_layer(base_dim, base_dim, base_dim * 4, repeats[0], starting=True)
#         self.conv3 = make_layer(base_dim * 4, base_dim * 2, base_dim * 8, repeats[1])
#         self.conv4 = make_layer(base_dim * 8, base_dim * 4, base_dim * 16, repeats[2])
#         self.conv5 = make_layer(base_dim * 16, base_dim * 8, base_dim * 32, repeats[3])
#
#         # 3번
#         self.avgpool = nn.AvgPool3d(kernel_size=7, stride=1)
#         self.classifer = nn.Linear(2048, self.num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.avgpool(x)
#         # 3번 2048x1 -> 1x2048
#         x = x.view(x.size(0), -1)
#         x = self.classifer(x)
#         return x

# ============
# class ResBlock(nn.Module):
# 	def __init__(self, block):
# 		super().__init__()
# 		self.block = block
# 	def forward(self, x):
# 		return self.block(x) + x #f(x) + x
#
#
# class ResBlock(nn.Module):
#     def __init__(self, nf):
#         super().__init__()
#         self.conv1 = conv_layer(nf, nf)
#         self.conv2 = conv_layer(nf, nf)
#
#     def forward(self, x): 반환
#
#     x + self.conv2(self.conv1(x))
#
# class Conv6Res(nn.Module):
#   def __init__(self, num_classes=5):
#     super().__init__()
#     self.name = 'conv6res'
#     self.model = nn.Sequential(
#         nn.Conv3d(3, 8, 3),
#         nn.ReLU(),
#         nn.BatchNorm3d(8),
#         nn.MaxPool3d(2),
#         ResBlock(
#             nn.Sequential(
#                 nn.Conv3d(8, 32, 3),
#                 nn.ReLU(),
#                 nn.BatchNorm3d(32),
#                 nn.MaxPool3d(2),
#                 nn.Conv3d(32, 64, 3),  # 32- 64
#                 nn.ReLU(),
#                 nn.BatchNorm3d(64),
#                 nn.MaxPool3d(2))),
#         ResBlock(
#             nn.Sequential(
#                 nn.Conv3d(64, 128, 3),  # 64 - 128
#                 nn.ReLU(),
#                 nn.BatchNorm3d(128))),
#
#         nn.Flatten(),
#         # nn.Linear(32 * 32 * 32, 256),
#         nn.MaxPool3d((1, 7, 7)),
#         nn.ReLU())
#
#
#     self.classifier = nn.Linear(512, num_classes)

#   def forward(self, x):
#       batch_size = x.size(0)
#       x = self.model(x)
#       x = x.view(batch_size, -1)
#       x = self.classifier(x)
#       return x
#
# # ==========================
# class Conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
#         super(Conv_block, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
#         self.batchnorm = nn.BatchNorm2d(out_channels)
#         self.activation = activation
#
#     def forward(self, x):
#         if not self.activation:
#             return self.batchnorm(self.conv(x))
#         return self.relu(self.batchnorm(self.conv(x)))
#
#
# class Res_block(nn.Module):
#     def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
#         super(Res_block, self).__init__()
#         self.relu = nn.ReLU()
#         self.is_plain = is_plain
#
#         if in_channels == 64:
#             self.convseq = nn.Sequential(
#                 Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
#                 Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
#                 Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
#             )
#             self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         elif in_channels == out_channels:
#             self.convseq = nn.Sequential(
#                 Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
#                 Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
#                 Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
#             )
#             self.iden = nn.Identity()
#         else:
#             self.convseq = nn.Sequential(
#                 Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
#                 Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
#                 Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
#
#             )
#             self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
#
#     def forward(self, x):
#         y = self.convseq(x)
#         if self.is_plain:
#             x = y
#         else:
#             x = y + self.iden(x)
#         x = self.relu(x)  # relu(skip connection)
#         return x
#
#
#
# class ResNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=5, is_plain=False):
#         self.num_classes = num_classes
#         super(ResNet, self).__init__()
#         self.conv1 = Conv_block(in_channels=in_channels, out_channels=5, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.conv2_x = nn.Sequential(
#             Res_block(64, 64, 256, is_plain),
#             Res_block(256, 64, 256, is_plain),
#             Res_block(256, 64, 256, is_plain)
#         )
#
#         self.conv3_x = nn.Sequential(
#             Res_block(256, 128, 512, is_plain),
#             Res_block(512, 128, 512, is_plain),
#             Res_block(512, 128, 512, is_plain),
#             Res_block(512, 128, 512, is_plain)
#         )
#
#         self.conv4_x = nn.Sequential(
#             Res_block(512, 256, 1024, is_plain),
#             Res_block(1024, 256, 1024, is_plain),
#             Res_block(1024, 256, 1024, is_plain),
#             Res_block(1024, 256, 1024, is_plain),
#             Res_block(1024, 256, 1024, is_plain),
#             Res_block(1024, 256, 1024, is_plain)
#         )
#
#         self.conv5_x = nn.Sequential(
#             Res_block(1024, 512, 2048, is_plain),
#             Res_block(2048, 512, 2048, is_plain),
#             Res_block(2048, 512, 2048, is_plain),
#         )
#
#         self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
#         self.fc = nn.Linear(2048, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2_x(x)
#         x = self.conv3_x(x)
#         x = self.conv4_x(x)
#         x = self.conv5_x(x)
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

# class ResidualBlock(nn.Module):
# 	def __init__(self, block):
# 		super().__init__()
# 		self.block = block
# 	def forward(self, x):
# 		return self.block(x) + x #f(x) + x
#
# class ResNet50_layer4(nn.Module):
#     def __init__(self, num_classes= 10 ):
#         super(ResNet50_layer4, self).__init__()
#         self.layer1 = nn.Sequential(
#             # nn.Conv2d(3, 64, 7, 2, 3),
#             nn.Conv3d(3, 64, (3, 3, 3)),
#             nn.BatchNorm3d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(3)
#         )
#         self.layer2 = nn.Sequential(
#             ResidualBlock(64, 64, 256, False),
#             ResidualBlock(256, 64, 256, False),
#             ResidualBlock(256, 64, 256, True)
#         )
#         self.layer3 = nn.Sequential(
#             ResidualBlock(256, 128, 512, False),
#             ResidualBlock(512, 128, 512, False),
#             ResidualBlock(512, 128, 512, False),
#             ResidualBlock(512, 128, 512, True)
#         )
#         self.layer4 = nn.Sequential(
#             ResidualBlock(512, 256, 1024, False),
#             ResidualBlock(1024, 256, 1024, False),
#             ResidualBlock(1024, 256, 1024, False),
#             ResidualBlock(1024, 256, 1024, False),
#             ResidualBlock(1024, 256, 1024, False),
#             ResidualBlock(1024, 256, 1024, True)
#         )
#         self.layer5 = nn.Sequential(
#             ResidualBlock(1024, 512, 2048, False),
#             ResidualBlock(2048, 512, 2048, False),
#             ResidualBlock(2048, 512, 2048, False)
#         )
#         self.fc = nn.Linear(2048, 10)
#         self.avgpool = nn.AvgPool2d((2, 2), stride=0)
#
#         def forward(self, x):
#             out = self.layer1(x)
#             out = self.layer2(out)
#             out = self.layer3(out)
#             # out = self.layer4(out)
#             out = self.layer5(out)
#             out = self.avgpool(out)
#             out = out.view(out.size()[0], -1)
#             out = self.fc(out)
#             return out
#
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

    return best_model

# 분류 모델 평가
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

# model = Conv6Res().to(device)
# model = ResNet().to(device)
# model = ResNet50_layer4().to(device)
# model = basela().to(device)
# model = ResNet().to(device)

# model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['path'].values, None,train_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv('./baseline_submit.csv', index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))
# !pip install torch torchvision pandas scikit-learn matplotlib tables pyarrow

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import math
from collections import defaultdict, OrderedDict
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, mean_absolute_error, matthews_corrcoef

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

store = pd.HDFStore('/public/home/jd_ylf/ukdale3.h5')

def resample_meter(store=None, building=1, meter=1, period='1min', cutoff=1000.):
    key = '/building{}/elec/meter{}'.format(building,meter)
    m = store[key]
    v = m.values.flatten()
    t = m.index
    s = pd.Series(v, index=t).clip(0.,cutoff)
    s[s<10.] = 0.
    return s.resample('1s').ffill(limit=300).fillna(0.).resample(period).mean().tz_convert('UTC')

def get_series(datastore, house, label, cutoff):
    filename = r'/public/home/jd_ylf/anaconda3_new/envs/nilmtk-env/lib/python3.6/site-packages/nilmtk/disaggregate/transferNILM-master/dataset_management/ukdale/uk2015/house_%1d/labels.dat' %house
    print(filename)
    labels = pd.read_csv(filename, delimiter=' ', header=None, index_col=0).to_dict()[1]
    
    for i in labels:
        if labels[i] == label:
            print(i, labels[i])
            s = resample_meter(store, house, i, '1min', cutoff)
            #s = resample_meter(store, house, i, '6s', cutoff)
    
    s.index.name = 'datetime'
    
    return s

# house = 1
# m = get_series(store, house, 'aggregate', 10000.)
# m.name = 'aggregate'
# a1 = get_series(store, house, 'kettle', 3100.)
# a1.name = 'kettle'
# a2 = get_series(store, house, 'fridge', 300.)
# a2.name = 'fridge'
# a3 = get_series(store, house, 'washing_machine', 2500.)
# a3.name = 'washing_machine'
# a4 = get_series(store, house, 'microwave', 3000.)
# a4.name = 'microwave'
# a5 = get_series(store, house, 'dishwasher', 2500.)
# a5.name = 'dish_washer'
# ds_1 = pd.concat([m, a1, a2, a3, a4, a5], axis=1)
# ds_1.fillna(method='pad', inplace=True)
#
# ds_1_train = ds_1[pd.datetime(2013,4,12):pd.datetime(2014,12,15)]
# ds_1_valid = ds_1[pd.datetime(2014,12,15):]
#
# house = 2
# m = get_series(store, house, 'aggregate', 10000.)
# m.name = 'aggregate'
# a1 = get_series(store, house, 'kettle', 3100.)
# a1.name = 'kettle'
# a2 = get_series(store, house, 'fridge', 300.)
# a2.name = 'fridge'
# a3 = get_series(store, house, 'washing_machine', 2500.)
# a3.name = 'washing_machine'
# a4 = get_series(store, house, 'microwave', 3000.)
# a4.name = 'microwave'
# a5 = get_series(store, house, 'dish_washer', 2500.)
# a5.name = 'dish_washer'
# ds_2 = pd.concat([m, a1, a2, a3, a4, a5], axis=1)
# ds_2.fillna(method='pad', inplace=True)
#
# ds_2_train = ds_2[pd.datetime(2013,5,22):pd.datetime(2013,10,3,6,16)]
# ds_2_valid = ds_2[pd.datetime(2013,10,3,6,16):]
#
# house = 3
# m = get_series(store, house, 'aggregate', 10000.)
# m.name = 'aggregate'
# a1 = get_series(store, house, 'kettle', 3100.)
# a1.name = 'kettle'
# a2 = 0.*m
# a2.name = 'fridge'
# a3 = 0.*m
# a3.name = 'washing_machine'
# a4 = 0.*m
# a4.name = 'microwave'
# a5 = 0.*m
# a5.name = 'dish_washer'
# ds_3 = pd.concat([m, a1, a2, a3, a4, a5], axis=1)
# ds_3.fillna(method='pad', inplace=True)
#
# start = pd.to_datetime('2013-2-27').tz_localize('UTC')
# end = pd.to_datetime('2013-4-1 06:15:00').tz_localize('UTC')
#
# ds_3_train = ds_3[start:end]
# ds_3_valid = ds_3[pd.datetime(2013,4,1,6,15):]
#
#
# house = 4
# m = get_series(store, house, 'aggregate', 10000.)
# m.name = 'aggregate'
# a1 = get_series(store, house, 'kettle_radio', 3100.)
# a1.name = 'kettle'
# a2 = get_series(store, house, 'freezer', 300.)
# a2.name = 'fridge'
# a3 = 0.*m
# a3.name = 'washing_machine'
# a4 = 0.*m
# a4.name = 'microwave'
# a5 = 0.*m
# a5.name = 'dish_washer'
# ds_4 = pd.concat([m, a1, a2, a3, a4, a5], axis=1)
# ds_4.fillna(method='pad', inplace=True)
#
# # start4 = pd.datetime(2013,3,9)
# # end4 = pd.datetime(2013,9,24,6,15)
# start4 = pd.to_datetime('2013-3-9').tz_localize('UTC')
# end4 = pd.to_datetime('2013-9-24 06:15:00').tz_localize('UTC')
#
# ds_4_train = ds_4[start4:end4]
# ds_4_valid = ds_4[pd.datetime(2013,9,24,6,15):]
#
#
# house = 5
# m = get_series(store, house, 'aggregate', 10000.)
# m.name = 'aggregate'
# a1 = get_series(store, house, 'kettle', 3100.)
# a1.name = 'kettle'
# a2 = get_series(store, house, 'fridge_freezer', 300.)
# a2.name = 'fridge'
# a3 = get_series(store, house, 'washer_dryer', 2500.)
# a3.name = 'washing_machine'
# a4 = get_series(store, house, 'microwave', 3000.)
# a4.name = 'microwave'
# a5 = get_series(store, house, 'dishwasher', 2500.)
# a5.name = 'dish_washer'
# ds_5 = pd.concat([m, a1, a2, a3, a4, a5], axis=1)
# ds_5.fillna(method='pad', inplace=True)
#
# start5 = pd.to_datetime('2014-6-29').tz_localize('UTC')
# end5 = pd.to_datetime('2014-9-1').tz_localize('UTC')
#
# ds_5_train = ds_5[start5:end5]
# ds_5_valid = ds_5[pd.datetime(2014,9,1):]
#
#
# ds_1_train.reset_index().to_feather('./UKDALE_1_train.feather')
# ds_2_train.reset_index().to_feather('./UKDALE_2_train.feather')
# ds_3_train.reset_index().to_feather('./UKDALE_3_train.feather')
# ds_4_train.reset_index().to_feather('./UKDALE_4_train.feather')
# ds_5_train.reset_index().to_feather('./UKDALE_5_train.feather')
#
# ds_1_valid.reset_index().to_feather('./UKDALE_1_valid.feather')
# ds_2_valid.reset_index().to_feather('./UKDALE_2_valid.feather')
# ds_3_valid.reset_index().to_feather('./UKDALE_3_valid.feather')
# ds_4_valid.reset_index().to_feather('./UKDALE_4_valid.feather')
# ds_5_valid.reset_index().to_feather('./UKDALE_5_valid.feather')


# Read the feather dataframe resampled
def get_status(app, threshold, min_off, min_on):
    condition = app > threshold
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    on_events = idx[:,0].copy()
    off_events = idx[:,1].copy()
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000.)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        assert len(on_events) == len(off_events)

        on_duration = off_events - on_events
        on_events = on_events[on_duration > min_on]
        off_events = off_events[on_duration > min_on]

    s = app.copy()
    #s.iloc[:] = 0.
    s[:] = 0.

    for on, off in zip(on_events, off_events):
        #s.iloc[on:off] = 1.
        s[on:off] = 1.
    
    return s
    

class Power(data.Dataset):
    def __init__(self, meter=None, appliance=None, status=None, 
                 length=256, border=680, max_power=1., train=False):
        self.length = length
        self.border = border
        self.max_power = max_power
        self.train = train

        self.meter = meter.copy()/self.max_power
        self.appliance = appliance.copy()/self.max_power
        self.status = status.copy()

        self.epochs = (len(self.meter) - 2*self.border) // self.length
        
    def __getitem__(self, index):
        i = index * self.length + self.border
        if self.train:
            i = np.random.randint(self.border, len(self.meter) - self.length - self.border)

        x = self.meter.iloc[i-self.border:i+self.length+self.border].values.astype('float32')
        y = self.appliance.iloc[i:i+self.length].values.astype('float32')
        s = self.status.iloc[i:i+self.length].values.astype('float32')
        x -= x.mean()
        
        return x, y, s

    def __len__(self):
        return self.epochs

        
class Encoder(nn.Module):
    def __init__(self, in_features=3, out_features=1, kernel_size=3, padding=1, stride=1):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        #return self.bn(F.relu(self.conv(x)))
        return self.drop(self.bn(F.relu(self.conv(x))))

class TemporalPooling(nn.Module):
    def __init__(self, in_features=3, out_features=1, kernel_size=2):
        super(TemporalPooling, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
        #self.upsample = nn.Upsample( scale_factor=kernel_size, mode='linear', align_corners=True)
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        #return self.upsample(x)
        #return self.drop(self.upsample(x))
        return self.drop(F.interpolate(x, scale_factor=self.kernel_size, mode='linear', align_corners=True))

class Decoder(nn.Module):
    def __init__(self, in_features=3, out_features=1, kernel_size=2, stride=2):
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return F.relu(self.conv(x))

class PTPNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(PTPNet, self).__init__()
        p = 2
        k = 1
        features = init_features
        self.encoder1 = Encoder(in_channels, features, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=p, stride=p)
        self.encoder2 = Encoder(features * 1**k, features * 2**k, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=p, stride=p)
        self.encoder3 = Encoder(features * 2**k, features * 4**k, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=p, stride=p)
        self.encoder4 = Encoder(features * 4**k, features * 8**k, kernel_size=3, padding=0)
        
        self.tpool1 = TemporalPooling(features*8**k, features*2**k, kernel_size=5)
        self.tpool2 = TemporalPooling(features*8**k, features*2**k, kernel_size=10)
        self.tpool3 = TemporalPooling(features*8**k, features*2**k, kernel_size=20)
        self.tpool4 = TemporalPooling(features*8**k, features*2**k, kernel_size=30)

        self.decoder = Decoder(2*features * 8**k, features * 1**k, kernel_size=p**3, stride=p**3)

        self.activation = nn.Conv1d(features * 1**k, out_channels, kernel_size=1, padding=0)

        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        tp1 = self.tpool1(enc4)
        tp2 = self.tpool2(enc4)
        tp3 = self.tpool3(enc4)
        tp4 = self.tpool4(enc4)

        dec = self.decoder(torch.cat([enc4, tp1, tp2, tp3, tp4], dim=1))

        act = self.activation(dec)
        return act

x = torch.randn(32,1,60*8+2*16)
model = PTPNet(1,3,32)
print(model(x).shape)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

######################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

# from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# net = UNet(n_channels=3, n_classes=1, bilinear=True)
x = torch.randn(32,1,60*8+2*16)
model = UNet(n_channels=1, n_classes=3, bilinear=True)
#print(model(x).shape)
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))




###############################################################
##### SegNet
bn_momentum = 0.1
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.enco1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        id = []

        x = self.enco1(x)
        x, id1 = F.max_pool1d(x, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)
        x, id2 = F.max_pool1d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        x = self.enco3(x)
        x, id3 = F.max_pool1d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        x = self.enco4(x)
        x, id4 = F.max_pool1d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        x = self.enco5(x)
        x, id5 = F.max_pool1d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)

        return x, id


# 编码器+解码器
class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)

        self.deco1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(64, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x, id = self.encoder(x)

        x = F.max_unpool1d(x, id[4], kernel_size=2, stride=2)
        x = self.deco1(x)
        x = F.max_unpool1d(x, id[3], kernel_size=2, stride=2)
        x = self.deco2(x)
        x = F.max_unpool1d(x, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        x = F.max_unpool1d(x, id[1], kernel_size=2, stride=2)
        x = self.deco4(x)
        x = F.max_unpool1d(x, id[0], kernel_size=2, stride=2)
        x = self.deco5(x)

        return x

#input = torch.randn(32,1,480)#.cuda()
#model = SegNet(1,3)
#model.eval()
#print(model)
#output = model(input)
#print('SegNet', output.size())

###################



def train_model(model, batch_size, n_epochs, filename):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the test loss as the model trains
    test_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    # to track the average test loss per epoch as the model trains
    avg_test_losses = [] 
    
    min_loss = np.inf
    
    # initialize the early_stopping object
    #patience = 10
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (data, target_power, target_status) in enumerate(train_loader, 1):
            data = data.unsqueeze(1).cuda()
            target_power = target_power.cuda()
            target_status = target_status.cuda()
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output_status = model(data).permute(0,2,1)
            # calculate the loss
            loss = criterion(output_status, target_status)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target_power, target_status in valid_loader:
            data = data.unsqueeze(1).cuda()
            target_power = target_power.cuda()
            target_status = target_status.cuda()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output_status = model(data).permute(0,2,1)
            # calculate the loss
            loss = criterion(output_status, target_status)
            # record validation loss
            valid_losses.append(loss.item())

        ##################    
        # test the model #
        ##################
        model.eval() # prep model for evaluation
        for data, target_power, target_status in test_loader:
            data = data.unsqueeze(1).cuda()
            target_power = target_power.cuda()
            target_status = target_status.cuda()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output_status = model(data).permute(0,2,1)
            # calculate the loss
            loss = criterion(output_status, target_status)
            # record validation loss
            test_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        test_loss = np.average(test_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_test_losses.append(test_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'test_loss: {test_loss:.5f} ')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        test_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        #early_stopping(valid_loss, model)
        #if (early_stopping.early_stop and (epoch > 80)):
        #    break
        
        if valid_loss < min_loss:
            print(f'Validation loss decreased ({min_loss:.6f} --> {valid_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), filename)
            min_loss = valid_loss
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(filename))
    
    return  model, avg_train_losses, avg_valid_losses, avg_test_losses

    
    
def evaluate_activation(model, loader, a):
    x_true = []
    s_true = []
    p_true = []
    s_hat = []
    
    model.eval()
    with torch.no_grad():
        for x, p, s in loader:
            x = x.unsqueeze(1).cuda()
            p = p.permute(0,2,1)[:,a,:]
            s = s.permute(0,2,1)[:,a,:]
            
            sh = model(x)
            sh = torch.sigmoid(sh[:,a,:])
            
            s_hat.append(sh.contiguous().view(-1).detach().cpu().numpy())
            
            x_true.append(x[:,:,BORDER:-BORDER].contiguous().view(-1).detach().cpu().numpy())
            s_true.append(s.contiguous().view(-1).detach().cpu().numpy())
            p_true.append(p.contiguous().view(-1).detach().cpu().numpy())
    x_true = np.hstack(x_true)
    s_true = np.hstack(s_true)
    p_true = np.hstack(p_true)
    s_hat = np.hstack(s_hat)

    return x_true, p_true, s_true, s_hat


APPLIANCE = ['fridge', 'dish_washer', 'washing_machine']
THRESHOLD = [50., 10., 20.]
MIN_ON = [1., 30., 30.]
MIN_OFF = [1., 30., 3.]

METER = 'aggregate'
SEQ_LEN = 60*8
BORDER = 0
BATCH_SIZE = 32

MAX_POWER = 2000.


ds_meter = []
ds_appliance = []
ds_status = []
for i in range(5):
    ds = pd.read_feather('./UKDALE_%d_train.feather' %(i+1))
    ds.set_index('datetime', inplace=True)
    
    meter = ds[METER]
    appliances = ds[APPLIANCE]
    
    status = pd.DataFrame()
    for a in range(len(APPLIANCE)):
        status = pd.concat([status, get_status(ds[APPLIANCE[a]], THRESHOLD[a], MIN_OFF[a], MIN_ON[a])], axis=1)
    
    ds_meter.append(meter)
    ds_appliance.append(appliances)
    ds_status.append(status)

ds_len = [len(ds_meter[i]) for i in range(5)]


(ds_status[1].diff()==1).sum()



ds_status[1].describe()



ds_house_train = [Power(ds_meter[i][:int(0.8*ds_len[i])], 
                        ds_appliance[i][:int(0.8*ds_len[i])], 
                        ds_status[i][:int(0.8*ds_len[i])], 
                        SEQ_LEN, BORDER, MAX_POWER, True) for i in range(5+0)]

ds_house_valid = [Power(ds_meter[i][int(0.8*ds_len[i]):int(0.9*ds_len[i])], 
                        ds_appliance[i][int(0.8*ds_len[i]):int(0.9*ds_len[i])],
                        ds_status[i][int(0.8*ds_len[i]):int(0.9*ds_len[i])], 
                        SEQ_LEN, BORDER, MAX_POWER, False) for i in range(5+0)]

ds_house_test  = [Power(ds_meter[i][int(0.9*ds_len[i]):], 
                        ds_appliance[i][int(0.9*ds_len[i]):],
                        ds_status[i][int(0.9*ds_len[i]):], 
                        SEQ_LEN, BORDER, MAX_POWER, False) for i in range(5+0)]

ds_house_total  = [Power(ds_meter[i], ds_appliance[i], ds_status[i], 
                         SEQ_LEN, BORDER, MAX_POWER, False) for i in range(5+0)]

ds_train_seen = torch.utils.data.ConcatDataset([ds_house_train[0], 
                                                ds_house_train[1], 
                                                #ds_house_train[2], 
                                                #ds_house_train[3],
                                                ds_house_train[4]
                                                ])
ds_valid_seen = torch.utils.data.ConcatDataset([ds_house_valid[0], 
                                                #ds_house_valid[1], 
                                                #ds_house_valid[2], 
                                                #ds_house_valid[3], 
                                                #ds_house_valid[4]
                                                ])

dl_train_seen = DataLoader(dataset = ds_train_seen, batch_size = BATCH_SIZE, shuffle=True)
dl_valid_seen = DataLoader(dataset = ds_valid_seen, batch_size = BATCH_SIZE, shuffle=False)
dl_test_seen = DataLoader(dataset = ds_house_test[0], batch_size = BATCH_SIZE, shuffle=False)

ds_train_unseen = torch.utils.data.ConcatDataset([ds_house_train[0], 
                                                  #ds_house_train[1], 
                                                  #ds_house_train[2], 
                                                  #ds_house_train[3], 
                                                  ds_house_train[4]
                                                  ])
ds_valid_unseen = torch.utils.data.ConcatDataset([ds_house_valid[0], 
                                                  #ds_house_valid[1], 
                                                  #ds_house_valid[2], 
                                                  #ds_house_valid[3], 
                                                  ds_house_valid[4]
                                                  ])
dl_train_unseen = DataLoader(dataset = ds_train_unseen, batch_size = BATCH_SIZE, shuffle=True)
dl_valid_unseen = DataLoader(dataset = ds_valid_unseen, batch_size = BATCH_SIZE, shuffle=False)
dl_test_unseen = DataLoader(dataset = ds_house_total[1], batch_size = BATCH_SIZE, shuffle=False)

dl_house_test = [DataLoader(dataset = ds_house_test[i], batch_size = 1, shuffle=False) for i in range(5)]
dl_house_valid = [DataLoader(dataset = ds_house_valid[i], batch_size = 1, shuffle=False) for i in range(5)]
dl_house_total = [DataLoader(dataset = ds_house_total[i], batch_size = 1, shuffle=False) for i in range(5)]

dataiter = iter(dl_house_test[1])


plt.figure(figsize=(15,8))
#x, y, s = dataiter.next()
a = 1
for i in range(100):
    x, y, s = dataiter.next()
    if y[0,:,a].sum() > 0:
        break
    if s[0,:,a].sum() > 0:
        break
plt.plot(np.arange(-BORDER, SEQ_LEN + BORDER), x[0,:].detach().numpy(), 'k-')
plt.plot(y[0,:,a].detach().numpy())
plt.plot(s[0,:,a].detach().numpy())
plt.ylim([-0.5,1.5])




# batch_size = BATCH_SIZE
# n_epochs = 100
#
# train_loader = dl_train_seen
# valid_loader = dl_valid_seen
# test_loader = dl_test_seen

#i = 0
# for i in range(10):
#     print('TRAINING MODEL %d' %i)
#     # Instantiate the model
#     model = PTPNet(1,3,32).cuda()
#     optimizer = optim.Adam(model.parameters(), lr=1.E-4)
#     criterion = nn.BCEWithLogitsLoss()
#     fn = 'UKDALE_seen_%d.pth' %i
#     model, train_loss, valid_loss, test_loss = train_model(model, batch_size, n_epochs, fn)
#
#
#
# plt.plot(train_loss)
# plt.plot(valid_loss)
# plt.plot(test_loss)
#
# plt.yscale('log')
# plt.grid(True)




batch_size = BATCH_SIZE
n_epochs = 70

train_loader = dl_train_unseen
valid_loader = dl_valid_unseen
test_loader = dl_test_unseen

#i = 0
for i in range(20):
    print('TRAINING MODEL %d' %i)
    # Instantiate the model
    #model = PTPNet(1,3,32).cuda()
    model = SegNet(1,3).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1.E-4)
    criterion = nn.BCEWithLogitsLoss()
    fn = './SegNet_model/SegNet_UKDALE_unseen_%d.pth' %i
    model, train_loss, valid_loss, test_loss = train_model(model, batch_size, n_epochs, fn)

  
#
#
#
plt.plot(train_loss)
plt.plot(valid_loss)
plt.plot(test_loss)

plt.yscale('log')
plt.grid(True)
plt.savefig('./SegNet_model/unseen_loss.png')
#
#
#model = PTPNet(1,3,32).cuda()
#print(model.eval())
# #
# scores = {}
# for a in range(3):
#     scores[a] = {}
#     scores[a]['F1'] = []
#     scores[a]['Precision'] = []
#     scores[a]['Recall'] = []
#     scores[a]['Accuracy'] = []
#     scores[a]['MCC'] = []
#     scores[a]['MAE'] = []
#     scores[a]['SAE'] = []
#
# thr = 0.5
# for i in range(10):
#     #filename = '/content/gdrive/My Drive/NILM/UKDALE_seen_%d.pth' %i
#     filename = './UKDALE_seen_%d.pth' %i
#     print(filename)
#     model.load_state_dict(torch.load(filename))
#     for a in range(3):
#         #x_true, p_true, s_true, s_hat = evaluate_activation(model, dl_house_total[0], a)
#         #pm = p_true.sum() / s_true.sum()
#         #pm = (ds_appliance[0][APPLIANCE[a]] *
#         #      ds_status[0][APPLIANCE[a]]).sum() / ds_status[0][APPLIANCE[a]].sum() / MAX_POWER
#         pm = ds_appliance[0][APPLIANCE[a]].sum() / ds_status[0][APPLIANCE[a]].sum() / MAX_POWER
#         x_true, p_true, s_true, s_hat = evaluate_activation(model, dl_house_test[0], a)
#         s_hat = get_status(s_hat, thr, MIN_OFF[a], MIN_ON[a])
#         p_hat = pm * s_hat
#         scores[a]['F1'].append(f1_score(s_true, s_hat))
#         scores[a]['Precision'].append(precision_score(s_true, s_hat))
#         scores[a]['Recall'].append(recall_score(s_true, s_hat))
#         scores[a]['Accuracy'].append(accuracy_score(s_true, s_hat))
#         scores[a]['MCC'].append(matthews_corrcoef(s_true, s_hat))
#         scores[a]['MAE'].append(mean_absolute_error(p_true, p_hat)*MAX_POWER)
#         scores[a]['SAE'].append((p_hat.sum() - p_true.sum()) / p_true.sum())
#
# for i,a in enumerate(APPLIANCE):
#     print()
#     print(a)
#     print('F1 score  : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['F1']), sorted(scores[i]['F1'])[1], sorted(scores[i]['F1'])[8]))
#     print('Precision : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Precision']), sorted(scores[i]['Precision'])[1], sorted(scores[i]['Precision'])[8]))
#     print('Recall    : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Recall']), sorted(scores[i]['Recall'])[1], sorted(scores[i]['Recall'])[8]))
#     print('Accuracy  : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Accuracy']), sorted(scores[i]['Accuracy'])[1], sorted(scores[i]['Accuracy'])[8]))
#     print('MCC       : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['MCC']), sorted(scores[i]['MCC'])[1], sorted(scores[i]['MCC'])[8]))
#     print('MAE       : %.2f (%.2f, %.2f)' %(np.mean(scores[i]['MAE']), sorted(scores[i]['MAE'])[1], sorted(scores[i]['MAE'])[8]))
#     print('SAE       : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['SAE']), sorted(scores[i]['SAE'])[1], sorted(scores[i]['SAE'])[8]))



    
    
scores = {}
for a in range(3):
    scores[a] = {}
    scores[a]['F1'] = []
    scores[a]['Precision'] = []
    scores[a]['Recall'] = []
    scores[a]['Accuracy'] = []
    scores[a]['MCC'] = []
    scores[a]['MAE'] = []
    scores[a]['SAE'] = []

thr = 0.5

for i in range(20):
    #filename = '/content/gdrive/My Drive/NILM/UKDALE_unseen_%d.pth' %i
    filename = './SegNet_model/SegNet_UKDALE_unseen_%d.pth' %i
    print(filename)
    model.load_state_dict(torch.load(filename))
    for a in range(3):
        x_true, p_true, s_true, s_hat = evaluate_activation(model, dl_house_total[1], a)
        pm = p_true.sum() / s_true.sum()
        pm = ds_appliance[1][APPLIANCE[a]].sum() / ds_status[1][APPLIANCE[a]].sum() / MAX_POWER
        x_true, p_true, s_true, s_hat = evaluate_activation(model, dl_house_total[1], a)
        s_hat = get_status(s_hat, thr, MIN_OFF[a], MIN_ON[a])
        p_hat = pm * s_hat
        scores[a]['F1'].append(f1_score(s_true, s_hat))
        scores[a]['Precision'].append(precision_score(s_true, s_hat))
        scores[a]['Recall'].append(recall_score(s_true, s_hat))
        scores[a]['Accuracy'].append(accuracy_score(s_true, s_hat))
        scores[a]['MCC'].append(matthews_corrcoef(s_true, s_hat))
        scores[a]['MAE'].append(mean_absolute_error(p_true, p_hat)*MAX_POWER)
        scores[a]['SAE'].append((p_hat.sum() - p_true.sum()) / p_true.sum())

for i,a in enumerate(APPLIANCE):
    print()
    print(a)
    print('F1 score  : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['F1']), sorted(scores[i]['F1'])[1], sorted(scores[i]['F1'])[18]))
    print('Precision : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Precision']), sorted(scores[i]['Precision'])[1], sorted(scores[i]['Precision'])[18]))
    print('Recall    : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Recall']), sorted(scores[i]['Recall'])[1], sorted(scores[i]['Recall'])[18]))
    print('Accuracy  : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['Accuracy']), sorted(scores[i]['Accuracy'])[1], sorted(scores[i]['Accuracy'])[18]))
    print('MCC       : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['MCC']), sorted(scores[i]['MCC'])[1], sorted(scores[i]['MCC'])[18]))
    print('MAE       : %.2f (%.2f, %.2f)' %(np.mean(scores[i]['MAE']), sorted(scores[i]['MAE'])[1], sorted(scores[i]['MAE'])[18]))
    print('SAE       : %.3f (%.3f, %.3f)' %(np.mean(scores[i]['SAE']), sorted(scores[i]['SAE'])[1], sorted(scores[i]['SAE'])[18]))



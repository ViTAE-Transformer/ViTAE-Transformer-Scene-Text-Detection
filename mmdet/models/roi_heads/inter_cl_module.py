from torch import nn
import torch
import torch.nn.functional as F


class GCGBlock(nn.Module):
    def __init__(self):
        super(GCGBlock, self).__init__()

        self.channels=256
        self.conv1x1_list = nn.ModuleList()
        for i in range(4 ):
            self.conv1x1_list.append(nn.Conv2d(self.channels, self.channels, 1, padding=1, bias=False))

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        for i in range(4):
            x[i] = self.conv1x1_list[i](x[i])

        global_context = self.globalavgpool(x[0])
        for j in range(1, 4):
            global_context += self.globalavgpool(x[j])

        return global_context


class GCTBlock(nn.Module):
    def __init__(self):
        super(GCTBlock, self).__init__()

        self.channels=256
        self.conv1x1_list = nn.ModuleList()
        for i in range(4):
            self.conv1x1_list.append(nn.Conv2d(self.channels, self.channels, 1, padding=1, bias=False))

        self.conv3x3_list = nn.ModuleList()
        for i in range(4):
            self.conv3x3_list.append(nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False))

        encoder_layer_pixel = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_encoder_pixel = nn.TransformerEncoder(encoder_layer_pixel, num_layers=3)

        self.maxpool_pixel = nn.AdaptiveMaxPool2d((14,14))

    def forward(self, x):
        feature_select_indx = 2
        N,C,H,W = x[feature_select_indx].shape
        feature_shape = x[feature_select_indx].shape[-2:]
        feature_fuse = self.conv1x1_list[feature_select_indx](x[feature_select_indx])

        for i, feature in enumerate(x):
            if i != feature_select_indx:
                feature = F.interpolate(feature, size=feature_shape, mode='bilinear', align_corners=True)
                feature_fuse += self.conv1x1_list[i](feature)

        for i in range(4):
            feature_fuse = self.conv3x3_list[i](feature_fuse)

        global_context = self.maxpool_pixel(global_context).view(N,C,14*14).permute(1,0,2).permute(2,1,0)
        global_context = self.transformer_encoder_pixel(global_context).permute(2,1,0).permute(1,0,2).view(N,C,14,14)

        return global_context


class InterCLBlock(nn.Module):
    def __init__(self):
        super(InterCLBlock, self).__init__()

        self.channels = 256
        self.conv1x1_reduce_channel = nn.Conv2d(self.channels, self.channels//8, kernel_size=1, stride=1, padding=0)
        self.conv1x1_return_channel = nn.Conv2d(self.channels//8, self.channels, kernel_size=1, stride=1, padding=0)

        encoder_layer_instance = nn.TransformerEncoderLayer(d_model=32*3*3, nhead=4)
        self.transformer_encoder_instance = nn.TransformerEncoder(encoder_layer_instance, num_layers=3)

        self.maxpool_instance = nn.AdaptiveMaxPool2d((3,3))


        self.conv3x3_out = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv1x1_out = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

        self.global_context_extract_module = GCGBlock()


    def forward(self, feature_map, roi_features):
        if roi_features.shape[0] == 0:
            return roi_features

        n,c,h,w = roi_features.shape

        roi_features_intercl = self.maxpool_instance(self.conv1x1_reduce_channel(roi_features)).view(n,1,(c//8)*3*3)
        
        roi_features_intercl = self.transformer_encoder_instance(roi_features_intercl).view(n,(c//8),3,3)
        roi_features_intercl = self.conv1x1_return_channel(roi_features_intercl)
        roi_features_intercl = F.interpolate(roi_features_intercl, size=[14,14], mode='bilinear', align_corners=True)

        global_context = self.global_context_extract_module(feature_map)

        roi_features = roi_features + roi_features_intercl + global_context
        roi_features = self.conv1x1_out(self.conv3x3_out(roi_features))
        roi_features = self.bn(roi_features)
        roi_features = self.relu(roi_features)

        return roi_features

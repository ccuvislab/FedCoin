'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
#             nn.Linear(25088, 256),  ### 1 for 224x224, cyjui.
            nn.Linear(4608, 256),  ### 1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),    ### 4
            nn.ReLU(True),
            nn.Linear(128, num_classes), ###6
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
#         print(self.features)
#         print("===== forward shape 0: {:}".format(x.shape))
        x = self.features(x)
#         print("===== forward shape 1: {:}".format(x.shape))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
class VGG_no_FC(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG_no_FC, self).__init__()
        self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(25088, 256),  ### 1 for 224x224, cyjui.
# #             nn.Linear(4608, 256),  ### 1
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(256, 128),    ### 4
#             nn.ReLU(True),
#             nn.Linear(128, num_classes), ###6
#         )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
#         print("===== forward shape 0: {:}".format(x.shape))
        x = self.features(x)
#         print("===== forward shape 1: {:}".format(x.shape))
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #print("in_channels: {}, v: {}".format(in_channels, v))
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                
            in_channels = v
            
    return nn.Sequential(*layers)


class VGGConvBlocks(nn.Module):
    '''
    VGG containers that only contains the conv layers 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class VGGContainer(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, input_dim, hidden_dims, num_classes=10):
        super(VGGContainer, self).__init__()
        self.features = features
        # note: we hard coded here a bit by assuming we only have two hidden layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[1], num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class VGGContainer_no_FC(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGGContainer_no_FC, self).__init__()
        self.features = features
        # note: we hard coded here a bit by assuming we only have two hidden layers
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(input_dim, hidden_dims[0]),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.ReLU(True),
#             nn.Linear(hidden_dims[1], num_classes),
#         )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        return x    


    
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
#     'ck':[64, 64, 'M', 128, 128, 'M', 256, 259, 259, 'M', 515, 515, 515, 'M', 515, 515, 512, 'M'],
    'ck':[64, 64, 'M', 128, 128, 'M', 256, 259, 259, 'M', 515, 515, 515, 'M', 515, 515, 512,],
#     'ck': [64, 64, 'M', 128, 128, 'M', 256, 259, 259, 'M', 515, 515, 512, 'M', 512, 512, 512],
    'skf':[64, 64, 'M', 128, 128, 'M', 256, 262, 262, 'M', 518, 518, 518, 'M', 518, 518, 512, 'M'],
    'ck-l6':[64, 64, 'M', 128, 128, 'M', 256, 259, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
    
def matched_vgg11(matched_shapes):
    # [(67, 27), (67,), (132, 603), (132,), (260, 1188), (260,), (261, 2340), (261,), (516, 2349), (516,), (517, 4644), (517,), 
    # (516, 4653), (516,), (516, 4644), (516,), (516, 515), (515,), (515, 515), (515,), (515, 10), (10,)]
    processed_matched_shape = [matched_shapes[0][0],  ### 0
                                'M', 
                                matched_shapes[2][0], ### 3
                                'M', 
                                matched_shapes[4][0], ### 6
                                matched_shapes[6][0], ### 8
                                'M', 
                                matched_shapes[8][0], ### 11
                                matched_shapes[10][0], ### 13
                                'M', 
                                matched_shapes[12][0], ### 16
                                matched_shapes[14][0], ### 18
                                'M'
                              ]
    
    # cyjui.
    print("In matched_vgg11: {:}, {:}, {:}".format(matched_shapes[16], 
                                                   matched_shapes[18], 
                                                   matched_shapes[20]))
    
#     CLASSIFIER_IN_DIM = matched_shapes[14][1]
    
    return VGGContainer(make_layers(processed_matched_shape), input_dim=matched_shapes[16][0], 
            hidden_dims=[matched_shapes[16][1], matched_shapes[18][1]], num_classes=10)
    
#     return VGGContainer(make_layers(processed_matched_shape), input_dim=matched_shapes[16][0], 
#             hidden_dims=[matched_shapes[16][1], matched_shapes[18][1]], num_classes=10)

def matched_vgg16(matched_shapes):
    # [(67, 27), (67,), (132, 603), (132,), (260, 1188), (260,), (261, 2340), (261,), (516, 2349), (516,), (517, 4644), (517,), 
    # (516, 4653), (516,), (516, 4644), (516,), (516, 515), (515,), (515, 515), (515,), (515, 10), (10,)]
    processed_matched_shape = [matched_shapes[0][0],  ### 0
                               matched_shapes[2][0],  ### 1
                               'M',
                               matched_shapes[4][0],  ### 2
                               matched_shapes[6][0],  ### 3
                               'M', 
                               matched_shapes[8][0],  ### 4
                               matched_shapes[10][0], ### 5
                               matched_shapes[12][0], ### 6
                               'M', 
                               matched_shapes[14][0], ### 7
                               matched_shapes[16][0], ### 8
                               matched_shapes[18][0], ### 9
                               'M', 
                               matched_shapes[20][0], ### 10
                               matched_shapes[22][0], ### 11
                               matched_shapes[24][0], ### 12
                               'M'
                              ]
    
#     # cyjui.
#     print("In matched_vgg16: {:}, {:}, {:}".format(matched_shapes[16], 
#                                                    matched_shapes[18], 
#                                                    matched_shapes[20]))
    
#     CLASSIFIER_IN_DIM = matched_shapes[14][1]

#     make_layers(processed_matched_shape)
    
#     print("mak_layers OK???")
    
    return VGGContainer(make_layers(processed_matched_shape), input_dim=matched_shapes[26][0], 
            hidden_dims=[matched_shapes[26][1], matched_shapes[28][1]], num_classes=1000)


def matched_vgg16_no_FC(matched_shapes):
    # [(67, 27), (67,), (132, 603), (132,), (260, 1188), (260,), (261, 2340), (261,), (516, 2349), (516,), (517, 4644), (517,), 
    # (516, 4653), (516,), (516, 4644), (516,), (516, 515), (515,), (515, 515), (515,), (515, 10), (10,)]
    processed_matched_shape = [matched_shapes[0][0],  ### 0
                               matched_shapes[2][0],  ### 1
                               'M',
                               matched_shapes[4][0],  ### 2
                               matched_shapes[6][0],  ### 3
                               'M', 
                               matched_shapes[8][0],  ### 4
                               matched_shapes[10][0], ### 5
                               matched_shapes[12][0], ### 6
                               'M', 
                               matched_shapes[14][0], ### 7
                               matched_shapes[16][0], ### 8
                               matched_shapes[18][0], ### 9
                               'M', 
                               matched_shapes[20][0], ### 10
                               matched_shapes[22][0], ### 11
                               matched_shapes[24][0], ### 12
#                                'M'
                              ]
    
#     # cyjui.
#     print("In matched_vgg16: {:}, {:}, {:}".format(matched_shapes[16], 
#                                                    matched_shapes[18], 
#                                                    matched_shapes[20]))
    
#     CLASSIFIER_IN_DIM = matched_shapes[14][1]

#     make_layers(processed_matched_shape)
    
#     print("mak_layers OK???")
    
    return VGGContainer_no_FC(make_layers(processed_matched_shape))


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def vgg16_rcnn():
    """VGG 16-layer model (configuration "D")"""
    return VGG_no_FC(make_layers(cfg['D']))

def vgg16_fedma(domain_cfg='ck'):
    """VGG 16-layer model (configuration "D")"""
    return VGG_no_FC(make_layers(cfg[domain_cfg]))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))


if __name__ == "__main__":
    matched_shapes = [(67, 27), (67,), (132, 603), (132,), (260, 1188), (260,), (261, 2340), (261,), (516, 2349), (516,), (517, 4644), 
    (517,), (516, 4653), (516,), (516, 4644), (516,), (516, 515), (515,), (515, 515), (515,), (515, 10), (10,)]
    net = matched_vgg11(matched_shapes=matched_shapes)
    for k, v in net.state_dict().items():
        print("Key: {}, Weight shape: {}".format(k, v.size()))
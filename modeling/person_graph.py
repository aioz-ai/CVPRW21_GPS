"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
-------------------------------------------------------------------------------
Graph-based Signature (GPS)
Written by @author Binh X. Nguyen
Paper: Graph-based Person Signature for Person Re-Identifications
https://github.com/aioz-ai/CVPRW21_GPS
"""

from torch import nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.resnet_nl import ResNetNL
from torch.nn import Parameter
from utils.util import *


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def extract_part_features(feature_maps, part_mask):
    batch_size, channel, h, w = feature_maps.shape
    x = F.interpolate(input=feature_maps, size=((h, w)), mode='bilinear', align_corners=True)
    value = x.view(batch_size, channel, -1)
    value = value.permute(0, 2, 1)
    part_mask = F.interpolate(input=part_mask.type(torch.cuda.FloatTensor), size=((h, w)), mode='nearest')
    part_mask = F.normalize(part_mask, p=1, dim=2)
    part_mask = part_mask.view(batch_size, -1, w * h)
    part_feats = torch.matmul(part_mask, value)
    return part_feats


def get_attr_part_pair(dataset_name):
    if dataset_name == 'market1501':
        attr_order = ['young', 'teenager', 'adult', 'old', 'backpack', 'bag', 'handbag', 'clothes', 'down', 'up', 'hair',
                  'hat',
                  'gender', 'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen',
                  'downblack',
                  'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']

        attribute_dict = {
            'young': 'global',
            'teenager': 'global',
            'adult': 'global',
            'old': 'global',
            'backpack': 'upper',
            'bag': 'arm',
            'handbag': 'arm',
            'clothes': 'lower',
            'down': 'lower',
            'up': 'arm',
            'hair': 'head',
            'hat': 'head',
            'gender': 'global',
            'upblack': 'upper',
            'upwhite': 'upper',
            'upred': 'upper',
            'uppurple': 'upper',
            'upyellow': 'upper',
            'upgray': 'upper',
            'upblue': 'upper',
            'upgreen': 'upper',
            'downblack': 'lower',
            'downpink': 'lower',
            'downpurple': 'lower',
            'downyellow': 'lower',
            'downgray': 'lower',
            'downblue': 'lower',
            'downgreen': 'lower',
            'downbrown': 'lower',
            'downwhite': 'lower'
        }
    elif dataset_name == 'dukemtmc':
        attr_order = ['backpack', 'bag', 'handbag', 'boots', 'gender', 'hat', 'shoes', 'top', 'upblack', 'upwhite', 'upred',
        'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown', 'downblack', 'downwhite', 'downred', 'downgray',
        'downblue', 'downgreen', 'downbrown']

        attribute_dict = {
            'backpack': 'upper',
            'bag': 'arm',
            'handbag': 'arm',
            'boots': 'lower',
            'gender': 'global',
            'hat': 'head',
            'shoes': 'lower',
            'top': 'upper',
            'upblack': 'upper',
            'upwhite': 'upper',
            'upred': 'upper',
            'uppurple': 'upper',
            'upgray': 'upper',
            'upblue': 'upper',
            'upgreen': 'upper',
            'upbrown': 'upper',
            'downblack': 'lower',
            'downwhite': 'lower',
            'downred': 'lower',
            'downgray': 'lower',
            'downblue': 'lower',
            'downgreen': 'lower',
            'downbrown': 'lower'
        }
    body_part_order = ['bg', 'head', 'upper', 'lower', 'arm', 'global']
    attribute_part_pair = [body_part_order.index(attribute_dict[i]) for i in attr_order]
    return attribute_part_pair

class PersonGraph(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, dataset_name, part, att):
        super(PersonGraph, self).__init__()
        # Select backbone
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride,
                                 block=Bottleneck,
                                 layers=[3, 4, 6, 3],
                                 non_layers=[0, 2, 3, 0])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        print("Global Adaptive Pooling")
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Number of identity
        self.num_classes = num_classes
        # Use BatchNorm Bottleneck
        self.neck = neck
        # Use neck feature for inference, default is 'on'
        self.neck_feat = neck_feat
        # Training with body parts
        self.part = part
        # Training with attributes
        self.att = att
        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes + 1024, self.num_classes)
        elif self.neck == 'bnneck':
            # Bottleneck layer
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes + 1024, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        # Init GCN with 300 dim input
        if part == 'on' or att == 'on':
            self.gc1 = GraphConvolution(300, 1024)
            self.gc2 = GraphConvolution(1024, 2048)
            self.relu = nn.LeakyReLU(0.2)

        if dataset_name == 'market1501':
            self.num_att = 30
            k = [(0, 34), (1, 34), (2, 34), (3, 34), (4, 31), (5, 33), (6, 33), (7, 32), (8, 32), (9, 33), (10, 30),
                 (11, 30), (12, 34), (13, 31), (14, 31), (15, 31), (16, 31), (17, 31), (18, 31), (19, 31), (20, 31),
                 (21, 32), (22, 32), (23, 32), (24, 32), (25, 32), (26, 32), (27, 32), (28, 32), (29, 32)]
        elif dataset_name == 'dukemtmc':
            self.num_att = 23
            k = [(0, 24), (1, 26), (2, 26), (3, 25), (4, 27), (5, 23), (6, 25), (7, 24), (8, 24), (9, 24), (10, 24),
                 (11, 24), (12, 24), (13, 24), (14, 24), (15, 24), (16, 25), (17, 25), (18, 25), (19, 25), (20, 25),
                 (21, 25), (22, 25)]
        # Generate correlation matrix from statistic attributes info and binirize
        _adj = gen_A(self.num_att, t = 0.9, adj_file='dataset/' + dataset_name + '/adj.pkl')
        # Add body part correlation to the correlation matrix
        if self.part == 'on':
            self.A = torch.zeros((self.num_att + 5, self.num_att + 5))
            self.A[:self.num_att, :self.num_att] = torch.from_numpy(_adj).float()
            for i in k:
                self.A[i[1], i[0]] = 1.0
                self.A[i[0], i[1]] = 1.0
            self.A[self.num_att:, self.num_att:] = torch.ones(5, 5)
            if self.att == 'off':
                self.A = torch.ones(5, 5)
            self.A = Parameter(self.A)
            self.convert_part = nn.Linear(2048, 300)
            self.attr_part_pair = get_attr_part_pair(dataset_name)
        elif self.part == 'off':
            if self.att == 'on':
                self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, x, inp, mask):
        global_feat = self.base(x) # (bs, 2048, 16, 8)
        # part representation here
        if self.part == 'on':
            part_feat = extract_part_features(global_feat, mask) # (bs, mask_channel, 2048)
            part_feat_conv = self.convert_part(part_feat[:, 1:, :]) # (bs, mask_channel, 300)
            if self.att == 'off':
                inp = part_feat_conv
            else:
                inp = torch.cat((inp, part_feat_conv), 1) # (bs, mask_channel + num_attributes, 300)
        elif self.part == 'off':
            inp = inp
        global_feat = self.gap(global_feat)  # (bs, 2048, 1, 1)
        # global feat used in triplet loss and center loss
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        # what output used in triplet loss?
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        adj = gen_adj(self.A).detach() # (30, 30)
        x = self.gc1(inp, adj) # (30, 1024)
        x_att = self.relu(x)
        x = self.gc2(x_att, adj) # (30, 2048)

        feat2 = feat.unsqueeze(2)
        att_feat = torch.matmul(x, feat2)
        att_feat = att_feat.squeeze(2)[:, :self.num_att]
        # (bs, 30)
        feat = torch.cat((feat, x_att.mean(dim=1)), dim=1)
        if self.training:
            cls_score = self.classifier(feat) # (bs, 751)
            return cls_score, global_feat, att_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.items():
            if 'classifier' in k:
                continue
            self.state_dict()[k].copy_(param_dict[k])



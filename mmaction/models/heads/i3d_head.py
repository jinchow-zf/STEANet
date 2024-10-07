import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [2, 768, 32, 1, 1]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score






####  for visualization only
# import torch
# import torch.nn as nn
# from mmcv.cnn import normal_init
# import numpy as np
# import os
#
# from ..builder import HEADS
# from .base import BaseHead
#
#
# @HEADS.register_module()
# class I3DHead(BaseHead):
#     """Classification head for I3D.
#
#     Args:
#         num_classes (int): Number of classes to be classified.
#         in_channels (int): Number of channels in input feature.
#         loss_cls (dict): Config for building loss.
#             Default: dict(type='CrossEntropyLoss')
#         spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
#         dropout_ratio (float): Probability of dropout layer. Default: 0.5.
#         init_std (float): Std value for Initiation. Default: 0.01.
#         feature_file (str): File path to save features. Default: 'features.txt'.
#         kwargs (dict, optional): Any keyword argument to be used to initialize
#             the head.
#     """
#
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  loss_cls=dict(type='CrossEntropyLoss'),
#                  spatial_type='avg',
#                  dropout_ratio=0.5,
#                  init_std=0.01,
#                  feature_file='/home/zj/Model_Code/AIM/Vislization_for_paper/VideoEmotion_tSNE/Baseline_add_TE_Adapter/features.txt',
#                  **kwargs):
#         super().__init__(num_classes, in_channels, loss_cls, **kwargs)
#
#         self.spatial_type = spatial_type
#         self.dropout_ratio = dropout_ratio
#         self.init_std = init_std
#         if self.dropout_ratio != 0:
#             self.dropout = nn.Dropout(p=self.dropout_ratio)
#         else:
#             self.dropout = None
#         self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
#
#         if self.spatial_type == 'avg':
#             # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
#             self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         else:
#             self.avg_pool = None
#
#         self.feature_file = feature_file
#         self.features_list = []
#
#     def init_weights(self):
#         """Initiate the parameters from scratch."""
#         normal_init(self.fc_cls, std=self.init_std)
#
#     def forward(self, x):
#         """Defines the computation performed at every call.
#
#         Args:
#             x (torch.Tensor): The input data.
#
#         Returns:
#             torch.Tensor: The classification scores for input samples.
#         """
#         # [N, in_channels, T, H, W]
#         if self.avg_pool is not None:
#             x = self.avg_pool(x)
#         # [N, in_channels, 1, 1, 1]
#         if self.dropout is not None:
#             x = self.dropout(x)
#         # [N, in_channels, 1, 1, 1]
#         x = x.view(x.shape[0], -1)
#         # [N, in_channels]
#
#         # Save features
#         self.save_features(x)
#
#         cls_score = self.fc_cls(x)
#         # [N, num_classes]
#         return cls_score
#
#     def save_features(self, x):
#         # Convert tensor to numpy array and add to list
#         features = x.detach().cpu().numpy()
#         self.features_list.extend(features.tolist())
#
#         # Write to file if accumulated features reach a threshold
#         if len(self.features_list) >= 100:  # Adjust this threshold as needed
#             self.write_features_to_file()
#
#     def write_features_to_file(self):
#         mode = 'a' if os.path.exists(self.feature_file) else 'w'
#         with open(self.feature_file, mode) as f:
#             for feature in self.features_list:
#                 f.write(str(feature).replace(' ', '') + ',\n')
#
#         # Clear the list
#         self.features_list = []
#
#     def __del__(self):
#         # Ensure all remaining features are written to file when object is destroyed
#         if self.features_list:
#             self.write_features_to_file()
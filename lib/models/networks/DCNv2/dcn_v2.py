'''
Author: Backlory
github: https://github.com/Backlory
Date: 2022-07-15 08:04:57
LastEditors: backlory's desktop dbdx_liyaning@126.com
LastEditTime: 2022-07-15 08:07:11
Description: 仿冒的DCN

Copyright (c) 2022 by Backlory, All Rights Reserved. 
'''
import torch.nn as nn

def DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1):
    return nn.Conv2d(chi, cho, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
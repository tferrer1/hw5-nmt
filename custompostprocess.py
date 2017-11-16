from __future__ import print_function
import utils.tensor
import utils.rand
import argparse
import dill
import logging
import sys
import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

with open('output.txt', 'rb') as file:
    for line in file:
        line_ = []
        token = False
        for word in line.split():
            line_.append(word)
        print(' '.join(line_).encode('utf-8').strip())

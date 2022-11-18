import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import sys
# sys.path.append('../')
from archs import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
# from datasets import MNIST, EMNIST, FashionMNIST

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        embedding_size = 2000
        output_size = embedding_size
        self.encoder = CNN_Encoder(output_size)

        self.decoder = CNN_Decoder(embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)
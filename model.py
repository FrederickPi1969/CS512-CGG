import networkx as nx
from utils import *
import torch.nn.functional as F
import torch
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from networkx.generators import random_graphs


class GCN(nn.Module):
    def __init__(self, output_dim, num_filters, device):
        super(GCN, self).__init__()
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.device = device
        self.bias = None
        self.kernel = None

    def forward(self, x, graph_conv_filters):
        # @param x : the graph attribute in a batch form
        assert graph_conv_filters.shape[-2] / graph_conv_filters.shape[-1] == self.num_filters
        if self.bias is None:
            self.bias = torch.randn(1).to(self.device)
            self.bias.requires_grad_()

        if self.kernel is None:
            self.kernel = torch.randn(x.shape[-1] * self.num_filters, self.output_dim).to(self.device)
            self.kernel.requires_grad_()

        # graph_conv_op shape of x should be 3-dimensional

        conv_op = torch.bmm(graph_conv_filters, x)
        conv_op = torch.split(conv_op, int(conv_op.shape[1] / self.num_filters), dim=1)
        print(len(conv_op), conv_op[0].shape)
        conv_op = torch.cat(conv_op, dim=2)

        conv_out = conv_op @ self.kernel
        conv_out += conv_out + self.bias  # optional
        conv_out = F.elu(conv_out)  # optional

        return conv_out


class Encoder(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(Encoder, self).__init__()
        self.num_filters = modelArgs["gnn_filters"]
        self.node_num = modelArgs["input_shape"][0][0]
        self.attr_dim = modelArgs["input_shape"][0][1]
        self.device = device

        self.gcn1 = GCN(100, self.num_filters, self.device) # fix output_dim = 100
        self.drop1 = nn.Dropout(0.1)
        self.gcn2 = GCN(100, self.num_filters, self.device)
        self.drop2 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(100, 8, bias=True)
        self.linear2 = nn.Linear(8,6, bias=True)
        self.mean_linear = nn.Linear(6, modelArgs["latent_dim"])
        self.log_var_linear = nn.Linear(6, modelArgs["latent_dim"])
        # graph_conv_filters = preprocess_adj_tensor_with_identity(torch.squeeze(A_train))

    def sampler(self, args):
        z_mean, z_log_var = args
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(1)[0]
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, input, graph_conv_filters):
        # input is attr
        x = self.gcn1(input, graph_conv_filters)
        x = self.drop1(x)
        x = self.gcn2(x, graph_conv_filters)
        x = self.drop2(x)
        x = torch.mean(x, dim=1) # node invariant layer
        print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        z_mean = self.mean_linear(x)
        z_log_var = self.log_var_linear(x)
        z = self.sampler((z_mean, z_log_var))
        return z


class Decoder(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(Decoder, self).__init__()
        self.device = device
        self.node_num = modelArgs["input_shape"][0][0]
        self.attr_dim = modelArgs["input_shape"][0][1]
        self.modelArgs = modelArgs
        self.trainArgs = trainArgs
        self.out_channels = modelArgs["conv_filters"]
        self.kernel_size = modelArgs["kernel_size"]
        self.latent_dim  = modelArgs["latent_dim"]

        # decoding A
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
        #                        kernel_size=self.kernel_size, stride=2,
        #                        padding=self.compute_same_padding_size(self.kernel_size))

        self.padding_size = self.compute_same_padding_size(self.kernel_size)
        self.conv2_w = self.compute_shape_after_conv(w=self.node_num, k=self.kernel_size, s=2,
                                                     p=self.compute_same_padding_size(self.kernel_size))

        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
        #                        kernel_size=self.kernel_size, stride=2,
        #                        padding=self.compute_same_padding_size(self.kernel_size))

        self.out_w = self.compute_shape_after_conv(w=self.conv2_w, k=self.kernel_size, s=2,
                                                   p=self.compute_same_padding_size(self.kernel_size))

        self.dense1 = nn.Linear(self.latent_dim, self.out_w * self.out_w * self.out_channels * 4)

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.out_channels * 4, out_channels= self.out_channels * 2,
                                          kernel_size=self.kernel_size, stride=2, padding=(self.padding_size[0], self.padding_size[1]))

        self.deconv2 = nn.ConvTranspose2d(in_channels= self.out_channels * 2, out_channels=self.out_channels,
                                          kernel_size=self.kernel_size, stride=2, padding=(self.padding_size[0], self.padding_size[1]))

        self.deconv3 = nn.ConvTranspose2d(in_channels= self.out_channels, out_channels=1,
                                          kernel_size=self.kernel_size, stride=2, padding=(self.padding_size[0], self.padding_size[1]))

        # decoding attribute
        self.linear1 = nn.Linear(self.latent_dim, 4)
        self.linear2 = nn.Linear(4, 6)
        self.linear3 = nn.Linear(6, 10)
        self.linear4 = nn.Linear(10, self.node_num)

    def compute_shape_after_conv(self, w, k, s, p):
        return (w + p[0] + p[1] - k) // s

    def compute_same_padding_size(self, kernel_size):
        two_p = kernel_size - 1
        p_left = int(two_p / 2)
        p_right = p_left if p_left == two_p / 2 else p_left + 1
        return (p_left, p_right, p_left, p_right)

    def forward(self, z):
        # decoding A:
        x = self.dense1(z)
        x = x.view(-1, self.out_channels * 4, self.out_w, self.out_w, )
        print(x.shape)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        A_hat = F.sigmoid(x).transpose(-1, -2).transpose(-1, 1)

        print(A_hat[0].squeeze(-1))
        # assert A_hat.transpose(1,2)[0].squeeze(-1) == A_hat[0].squeeze(-1)

        # decoding node attributes:
        y = self.linear1(z)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)
        y = F.relu(y)
        y = self.linear4(y)
        y = F.sigmoid(y)
        attr_hat = y.view(-1, self.node_num, self.attr_dim)

        return A_hat, attr_hat


class VAE(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(VAE, self).__init__()
        self.modelArgs = modelArgs
        self.trainArgs = trainArgs
        self.device = device
        self.encoder = Encoder(modelArgs, trainArgs, device)
        self.decoder = Decoder(modelArgs, trainArgs, device)

    def forward(self, attr, graph_conv_filters):
        z = self.encoder(attr, graph_conv_filters)
        A_hat, attr_hat = self.decoder(z)
        return A_hat, attr_hat








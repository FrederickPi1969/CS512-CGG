import networkx as nx
from utils import *
import torch.nn.functional as F
import torch
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, modelArgs, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.num_filters = modelArgs["gnn_filters"]
        self.node_num = modelArgs["input_shape"][0][0]
        self.attr_dim = modelArgs["input_shape"][0][1]

        self.gcn1 = GCN(64, self.num_filters, self.device)  # fix output_dim = 100
        self.drop1 = nn.Dropout(0.1)
        self.gcn2 = GCN(64, self.num_filters, self.device)

        self.linear1 = nn.Linear(64, 32, bias=True)
        self.linear2 = nn.Linear(32, 16, bias=True)
        self.linear3 = nn.Linear(16, 1, bias=True)

    def forward(self, x, graph_conv_filters):
        o = self.gcn1(x, graph_conv_filters)
        o = self.drop1(o)
        o = self.gcn2(o, graph_conv_filters)
        o = torch.mean(o, dim = 1) # mean
        # o = torch.max(o, dim = 1) # max pooling
        anchor = o
        o = self.linear1(o)
        o = F.leaky_relu(o)
        o = self.linear2(o)
        o = F.leaky_relu(o)
        o = self.linear3(o)
        o = torch.sigmoid(o)

        return anchor, o


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
            self.bias = torch.zeros(self.output_dim).to(self.device)
            self.bias.requires_grad_()

        if self.kernel is None:
            self.kernel = torch.nn.init.xavier_uniform_(torch.empty(x.shape[-1] * self.num_filters, self.output_dim), gain=1.0).to(self.device)
            self.kernel.requires_grad_()

        # graph_conv_op shape of x should be 3-dimensional

        conv_op = torch.bmm(graph_conv_filters, x)
        conv_op = torch.split(conv_op, int(conv_op.shape[1] / self.num_filters), dim=1)
#         # print(len(conv_op), conv_op[0].shape)
        conv_op = torch.cat(conv_op, dim=2)

        conv_out = conv_op @ self.kernel
        conv_out += conv_out + self.bias  # bias is optional!
        conv_out = F.elu(conv_out)  # activation is optional!

        return conv_out


class Encoder_v2(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(Encoder_v2, self).__init__()
        self.num_filters = modelArgs["gnn_filters"]
        self.node_num = modelArgs["input_shape"][0][0]
        self.attr_dim = modelArgs["input_shape"][0][1]
        self.device = device

        self.gcn1 = GCN(100, self.num_filters, self.device)  # fix output_dim = 100
        self.drop1 = nn.Dropout(0.1)
        self.gcn2 = GCN(100, self.num_filters, self.device)
        self.drop2 = nn.Dropout(0.1)

        self.node_emb_size = 128
        self.linear1 = nn.Linear(100, self.node_emb_size, bias=True)  # W1 size (h=100, emb=128)
        # self.linear1.bias.data.zero_()
        self.linear2 = nn.Linear(self.node_emb_size, 64, bias=True)  # W2 size (emb = 128, hidden=64)
        # self.linear2.bias.data.zero_()
        self.relu = nn.ReLU()
        self.mean_linear = nn.Linear(64, modelArgs["latent_dim"], bias=True)  # W3,4 size (emb=64, z)
        # self.mean_linear.bias.data.zero_()
        self.log_var_linear = nn.Linear(64, modelArgs["latent_dim"], bias=True)
        # self.log_var_linear.bias.data.zero_()

    def sampler(self, args):
        z_mean, z_log_var = args
        batch = z_mean.shape[0]
        dim = z_mean.shape[-1]
        n = z_mean.shape[1]
        epsilon = torch.randn(batch, n, dim).to(self.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, input, graph_conv_filters):
        # input is attr
        x = self.gcn1(input, graph_conv_filters)
        x = self.drop1(x)
        x = self.gcn2(x, graph_conv_filters)
        x = self.drop2(x)  # (b, n, h=100)
        # print(x.shape)

        x = self.relu(self.linear1(x))  # (b, n, emb_size)
        x = self.linear2(x)  # (b, n, hidden=64)

        z_mean = self.mean_linear(x)  # (b, n, z)
        z_log_var = self.log_var_linear(x)  # (b, n, z)
        z = self.sampler((z_mean, z_log_var))  # (b, n, z)
        return z_mean, z_log_var, z


class Decoder_v2(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(Decoder_v2, self).__init__()
        self.device = device
        self.node_num = modelArgs["input_shape"][0][0]
        self.attr_dim = modelArgs["input_shape"][0][-1]
        print(modelArgs["input_shape"])
        self.modelArgs = modelArgs
        self.trainArgs = trainArgs
        self.out_channels = modelArgs["conv_filters"]
        self.kernel_size = modelArgs["kernel_size"]
        self.latent_dim = modelArgs["latent_dim"]

        # decoding A
        # Input Z shape: (b, n, z_size) !
        self.hidden_shape = 64  # adjustable
        self.dense1 = nn.Linear(self.latent_dim, self.hidden_shape)
        # self.dense1_nm = nn.BatchNorm1d(self.conv2_w * self.conv2_w * self.out_channels * 4)

        # decoding attribute
        self.linear1 = nn.Linear(self.latent_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, self.attr_dim)

    def compute_shape_after_conv(self, w, k, s, p):
        return (w + p[0] + p[1] - k) // s + 1

    def compute_same_padding_size(self, kernel_size):
        two_p = kernel_size - 1
        p_left = int(two_p / 2)
        p_right = p_left if p_left == two_p / 2 else p_left + 1
        return (p_left, p_right, p_left, p_right)

    def forward(self, z):
        # # decoding A with dot product:
        x = self.dense1(z)  # (b, n, hidden=64)
        # x = self.dense1_nm(x)
        x = F.relu(x)
        A_hat = x.bmm(x.transpose(-1,-2)) # (b,n,n)
        max_score_per_node, _ = A_hat.max(dim=-1, keepdim=True)
        A_hat = (A_hat / (max_score_per_node + 1e-13)).clamp(0.02, 0.98).unsqueeze(-1)
        # print(A_hat)

        # decoding node attributes:
        y = self.linear1(z)  # (b, n, 32)
        y = F.relu(y)
        y = self.linear2(y)  # (b, n, 64)
        y = F.relu(y)
        y = self.linear3(y)  # (b, n, attr)
        attr_hat = torch.sigmoid(y)
        # attr_hat = y.view(-1, self.node_num, self.attr_dim)
        return A_hat, attr_hat


class VAE_v2(nn.Module):
    def __init__(self, modelArgs, trainArgs, device):
        super(VAE_v2, self).__init__()
        self.modelArgs = modelArgs
        self.trainArgs = trainArgs
        self.device = device
        self.encoder = Encoder_v2(modelArgs, trainArgs, device)
        self.decoder = Decoder_v2(modelArgs, trainArgs, device)

    def forward(self, attr, graph_conv_filters):
        z_mean, z_log_var, z = self.encoder(attr, graph_conv_filters)
        A_hat, attr_hat = self.decoder(z)

        return z_mean, z_log_var, z, A_hat, attr_hat

def binary_cross_entropy_loss(true, pred):
    # pred -= 1e-8
    # assert all(pred > 0.0)
    # assert all(pred < 1.0)
    loss = -1 * torch.mean(true * torch.log(pred) + (1 - true) * torch.log(1 - pred))
    if loss != loss:
        print(torch.max(pred), torch.min(pred))
        raise Exception("nan")
    return loss

def binary_cross_entropy_loss_w(true, pred):
    if true[0] == 1:
        return -1 * torch.mean(torch.log(pred))
    else:
        return -1 * torch.mean(torch.log(1 - pred))

def loss_func(y, y_hat, z_mean, z_log_var, trainArgs, modelArgs):
    A, A_hat = y[0], y_hat[0]
    attr, attr_hat = y[1], y_hat[1]
    mse = nn.MSELoss(reduction="mean")
    attr_reconstruction_loss = mse(attr.flatten(), attr_hat.flatten()) * modelArgs["input_shape"][0][0]

    bce = nn.BCELoss(reduction="mean")
    # adj_reconstruction_loss = bce(A.flatten(), A_hat.flatten().detach()) * (modelArgs["input_shape"][1][0] * modelArgs["input_shape"][1][1])
    # adj_reconstruction_loss.requires_grad_()

    # print(A_hat.flatten().requires_grad)

    # adj_reconstruction_loss = mse(A.flatten(), A_hat.flatten()) * (modelArgs["input_shape"][1][0] * modelArgs["input_shape"][1][1])
    adj_reconstruction_loss = binary_cross_entropy_loss(A.flatten(), A_hat.flatten()) * (modelArgs["input_shape"][1][0] * modelArgs["input_shape"][1][1])

    # print(torch.min(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()))
    kl_loss = -0.5 * torch.sum((1 + z_log_var - z_mean.pow(2) - z_log_var.exp()), dim = -1) ######## ??????????
    print(trainArgs["loss_weights"][0] * adj_reconstruction_loss, trainArgs["loss_weights"][1] * attr_reconstruction_loss, torch.mean(trainArgs["loss_weights"][2] * kl_loss))

    # print(trainArgs["loss_weights"][0] * adj_reconstruction_loss, trainArgs["loss_weights"][1] * attr_reconstruction_loss,  trainArgs["loss_weights"][2] * kl_loss)
    loss = torch.mean(trainArgs["loss_weights"][0] * adj_reconstruction_loss + trainArgs["loss_weights"][1] * attr_reconstruction_loss +  trainArgs["loss_weights"][2] * kl_loss)

    return loss


def w_loss_func(y, y_hat, feature_true, feature_fake, alpha, beta):
    mse = nn.MSELoss(reduction="mean")
    entropy_loss = binary_cross_entropy_loss_w(y.flatten(), y_hat.flatten())   ## modify w so as to maximize the probability of D being wrong!
    feature_similarity_loss = mse(feature_true, feature_fake)
    # return alpha * entropy_loss + beta * feature_similarity_loss
    return feature_similarity_loss




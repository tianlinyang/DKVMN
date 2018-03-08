import itertools
import json

import collections
import os
import re
import numpy as np
import codecs
import random
import torch.nn as nn
import torch.nn.init

def varible(tensor, gpu):
    if gpu >= 0 :
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def xavier_init(gpu, *size):
    return nn.init.xavier_normal(varible(torch.FloatTensor(*size), gpu))


def init_varaible_zero(gpu, *size):
    return varible(torch.zeros(*size), gpu)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)


def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_lstm_cell(input_lstm):
    weight = eval('input_lstm.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)
    weight = eval('input_lstm.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        weight = eval('input_lstm.bias_ih')
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        weight = eval('input_lstm.bias_hh')
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


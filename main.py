import torch
import numpy as np
from sklearn import metrics
import sys
import os, time, argparse
import logging
from model import MODEL
from run import train, test
import torch.optim as optim

from data_loader import DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=50, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')

    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int, default=20, help='memory size')

    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
    parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
    parser.add_argument('--data_dir', type=str, default='./data/assist2009_updated', help='data directory')
    parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
    parser.add_argument('--load', type=str, default='assist2009_updated', help='model file to load')
    parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')

    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    print(params)

    dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')

    seedNum=224
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim
    train_data_path = params.data_dir + "/" + params.data_name + "_train1.csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid1.csv"
    train_q_data, train_qa_data = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data = dat.load_data(valid_data_path)

    model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    final_fc_dim = params.final_fc_dim)

    model.init_params(params.init_std)

    optimizer = optim.SGD(params=model.parameters(), lr=params.lr, momentum=params.momentum)

    if params.gpus >= 0:
        if_cuda = True
        print('device: ' + str(params.gpus))
        torch.cuda.set_device(params.gpus)
        model.cuda()
    else:
        if_cuda = False

    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_q_data, train_qa_data)
        valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, valid_q_data, valid_qa_data)
        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy, "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc :
            best_valid_auc = valid_auc
            best_epoch = idx+1





if __name__ == "__main__":
    main()

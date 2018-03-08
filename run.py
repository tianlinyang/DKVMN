import numpy as np
import math
import torch
from torch import nn
import utils as utils
from sklearn import metrics

def train(model, params, optimizer, q_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):

        q_one_seq = q_data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_q = q_one_seq[:,:] # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx*params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        #target = target.astype(np.int)
        #print(target)
        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(input_q), params.gpus)
        input_qa = utils.varible(torch.LongTensor(input_qa), params.gpus)
        target = utils.varible(torch.FloatTensor(target), params.gpus)

        model.zero_grad()
        pred, loss = model.forward(input_q, input_qa, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(target.view(-1, 1).data.tolist())
        right_pred = np.asarray(pred.view(-1, 1).data.tolist())
        right_index = np.flatnonzero(right_target != -1.).tolist()
        pred_list.append(right_pred[right_index])
        target_list.append(right_target[right_index])


    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    utils.adjust_learning_rate(optimizer, params.init_lr / (1 + (20 + 1) * 0.667))
    print("all_target", all_target)
    print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc

def test(model, params, optimizer, q_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):

        q_one_seq = q_data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_q = q_one_seq[:,:] # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx*params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        #target = target.astype(np.int)
        #print(target)
        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(input_q), params.gpus)
        input_qa = utils.varible(torch.LongTensor(input_qa), params.gpus)
        target = utils.varible(torch.FloatTensor(target), params.gpus)

        pred, loss = model.forward(input_q, input_qa, target)

        right_target = np.asarray(target.view(-1, 1).data.tolist())
        right_pred = np.asarray(pred.view(-1, 1).data.tolist())
        right_index = np.flatnonzero(right_target != -1.).tolist()
        pred_list.append(right_pred[right_index])
        target_list.append(right_target[right_index])
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    # print("all_target", all_target)
    # print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc
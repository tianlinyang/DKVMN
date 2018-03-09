import torch
import torch.nn as nn
from memory import DKVMN
import utils as utils


class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim

        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.final_fc_dim, bias=True)
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim,
                                           bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.mem = DKVMN(memory_size=self.memory_size,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim, batch_size=self.batch_size)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim)

    def init_params(self, init_std):
        nn.init.uniform(self.q_embed.weight)
        nn.init.uniform(self.qa_embed.weight)
        # nn.init.normal(self.q_embed.weight, std=0.5)
        # nn.init.normal(self.qa_embed.weight, std=0.5)

    def forward(self, q_data, qa_data, target):
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = self.input_embed_linear(input_embed_content)
        input_embed_content = torch.tanh(input_embed_content)

        read_content_embed = self.read_embed_linear(torch.cat([all_read_value_content, input_embed_content], 2))
        read_content_embed = torch.tanh(read_content_embed)

        pred = self.predict_linear(read_content_embed)
        # target [batch_size, seq_len]
        # pred [batch_size, seq_len, 1]

        pred_logit = torch.sigmoid(pred)

        target_1d = target  # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)  # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)
        # loss = torch.nn.functional.binary_cross_entropy(filtered_pred, filtered_target)
        # torch.nn.functional.binary_cross_entropy()

        return pred_logit, loss, torch.sigmoid(filtered_pred), filtered_target

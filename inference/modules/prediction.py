import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        one_hot = input_char.new_zeros(input_char.size(0), onehot_dim)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, batch_max_length=25):
            return self._forward_inference(batch_H, batch_max_length)

    def _forward_train(self, batch_H, text, batch_max_length):
        num_steps = batch_max_length + 1

        output_hiddens = batch_H.new_zeros(batch_H.size(0), num_steps, self.hidden_size)
        hidden = (batch_H.new_zeros(batch_H.size(0), self.hidden_size),
                  batch_H.new_zeros(batch_H.size(0), self.hidden_size))

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
            output_hiddens[:, i, :] = hidden[0]
        probs = self.generator(output_hiddens)
        return probs

    def _forward_inference(self, batch_H, batch_max_length):
        num_steps = batch_max_length + 1

        hidden = (batch_H.new_zeros(batch_H.size(0), self.hidden_size),
                  batch_H.new_zeros(batch_H.size(0), self.hidden_size))
        targets = batch_H.new_zeros(batch_H.size(0), dtype=torch.long)

        probs_list = []
        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
            probs_step = self.generator(hidden[0])
            probs_list.append(probs_step.unsqueeze(1))
            _, next_input = probs_step.max(1)
            targets = next_input

        probs = torch.cat(probs_list, dim=1)
        return probs


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha

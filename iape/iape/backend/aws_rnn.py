import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNWDAbridgedModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers):
        super(RNNWDAbridgedModel, self).__init__()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhid = nhid
        self.nlayers = nlayers

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid, 1, dropout=0) for l in range(nlayers)]
        self.rnns = torch.nn.ModuleList(self.rnns)


    def forward(self, input, hidden, dones=None, breakpoints=None):

        emb = input
        #from tuple of multi_layer h,c to list of single layer h,c
        hidden =[(hidden[0][l].unsqueeze(0),hidden[1][l].unsqueeze(0)) for l in range(self.nlayers) ]

        if dones is None:
            raw_output = emb
            new_hidden = []
            # raw_outputs = []
            outputs = []
            for l, rnn in enumerate(self.rnns):
                current_input = raw_output
                raw_output, new_h = rnn(current_input, hidden[l])
                new_hidden.append(new_h)
                if l != self.nlayers - 1:
                    outputs.append(raw_output)
            hidden = new_hidden
        else:
            # raw_outputs = []
            outputs = []
            start_idx = 0
            final_raw_output=[]
            for end_idx in breakpoints:
                raw_output = emb[start_idx:end_idx+1]
                new_hidden = []
                expanded_continue = 1-dones[end_idx].unsqueeze(0).unsqueeze(-1)
                for l, rnn in enumerate(self.rnns):
                    current_input = raw_output
                    raw_output, new_h = rnn(current_input, hidden[l])
                    new_hidden.append((new_h[0]*expanded_continue, new_h[1]*expanded_continue))
                    if l != self.nlayers - 1:
                        outputs.append(raw_output)
                hidden = new_hidden
                final_raw_output.append(raw_output)
                start_idx=end_idx+1
            raw_output = torch.cat(final_raw_output,dim=0)
        #back to tuple
        h = torch.cat([h[0]for h in hidden],dim=0)
        c = torch.cat([h[1]for h in hidden],dim=0)
        hidden = (h,c)

        output = raw_output
        return output, hidden



    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid).zero_(),
                 weight.new(1, bsz, self.nhid).zero_())
                for l in range(self.nlayers)]

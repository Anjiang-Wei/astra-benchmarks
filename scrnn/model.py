import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 rnn_type,
                 ntoken,
                 ninp,
                 nhid,
                 nlayers,
                 dropout=0.5,
                 dropouth=0.5,
                 dropouti=0.5,
                 dropoute=0.1,
                 wdrop=0,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.RNN(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.drop(emb)

        raw_output = emb
        raw_outputs = []
        outputs = []
        raw_output, hidden = self.rnn(raw_output, hidden)
        raw_outputs.append(raw_output)

        output = self.drop(raw_output)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))
        return result, hidden, raw_outputs, outputs

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.nlayers, bsz, self.nhid, device='cuda')
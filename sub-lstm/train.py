import argparse
import time
import math
import numpy as np
import torch
import model
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument(
    '--data', type=str, default='penn_treebank_dataset', help='the name of the dataset to load')
parser.add_argument(
    '--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')
parser.add_argument(
    '--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument(
    '--dropouth', type=float, default=0.25, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--dropouti',
    type=float,
    default=0.4,
    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument(
    '--dropoute',
    type=float,
    default=0.1,
    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument(
    '--wdrop',
    type=float,
    default=0,
    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nonmono', type=int, default=5, help='random seed')
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument(
    '--save', type=str, default=randomhash + '.pt', help='path to save the final model')
parser.add_argument(
    '--alpha',
    type=float,
    default=2,
    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument(
    '--beta',
    type=float,
    default=1,
    help='beta slowness regularization applied on RNN activiation' +
    ' (beta = 0 means no regularization)')
parser.add_argument(
    '--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='', help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use (sgd, adam)')
parser.add_argument(
    '--when',
    nargs="+",
    type=int,
    default=[-1],
    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--torchscript', action='store_true', default=False, help='use TorchScript')
parser.add_argument('--trace', action='store_true', default=False, help='use jit trace')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


from torchnlp import datasets
from torchnlp.encoders import LabelEncoder
from torchnlp.samplers import BPTTBatchSampler

print('Producing dataset...')
train, val, test = getattr(datasets, args.data)(train=True, dev=True, test=True)

encoder = LabelEncoder(train + val + test)

train_data = encoder.batch_encode(train)
val_data = encoder.batch_encode(val)
test_data = encoder.batch_encode(test)

train_source_sampler, val_source_sampler, test_source_sampler = tuple(
    [BPTTBatchSampler(d, args.bptt, args.batch_size, True, 'source') for d in (train, val, test)])

train_target_sampler, val_target_sampler, test_target_sampler = tuple(
    [BPTTBatchSampler(d, args.bptt, args.batch_size, True, 'target') for d in (train, val, test)])

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = encoder.vocab_size
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = (args.dropouti, args.dropouth,
                                                                    args.dropout, args.dropoute)
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
model = model.to(device='cuda')
criterion = criterion.to(device='cuda')
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(
    x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


if args.torchscript:
    print("Scripting the module...")
    model = torch.jit.script(model)

if args.trace:
    example_input = torch.randn(args.bptt, args.batch_size).to('cuda')
    print('data.size():{}\n'.format(example_input.size()))
    print('type(data):{}\n'.format(type(example_input)))
    print('data:\n{}\n'.format(example_input))
    model = torch.jit.trace(model, example_input)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    total_loss = 0
    start_time = time.time()
    #hidden = model.init_hidden(args.batch_size)
    batch = 0
    for source_sample, target_sample in zip(train_source_sampler, train_target_sampler):
        model.train()
        data = torch.stack([train_data[i] for i in source_sample]).t_().contiguous().to(device='cuda')
        #print('data.size():{}\n'.format(data.size()))
        #print('type(data):{}\n'.format(type(data)))
        #print('data:\n{}\n'.format(data))
        targets = torch.stack([train_data[i] for i in target_sample]).t_().contiguous().view(-1).to(device='cuda')

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        #hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden, rnn_hs, dropped_rnn_hs = model(data)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                      epoch, batch,
                      len(train_source_sampler) // args.bptt,
                      optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss,
                      math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as
    # the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in tqdm(range(1, args.epochs + 1)):
        epoch_start_time = time.time()
        train()


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

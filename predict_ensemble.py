import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import NMT
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", default="data/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="words",
                    help="Source Language. (default = words)")
parser.add_argument("--trg_lang", default="phoneme",
                    help="Target Language. (default = phoneme)")
parser.add_argument("--model_file", default="model.py",
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=0.1, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the testelopment set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu testice to use. Empty implies cpu usage.")
parser.add_argument("--modelname", default="model.py.nll_0.68.epoch_18")
# feel free to add more arguments as you need


def to_var(input, volatile=True):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main(options):
  files = ['model_save.nll_0.73.epoch_15', 'model_save.nll_0.73.epoch_15', 'model_save.nll_0.73.epoch_15', 'model_save.nll_0.73.epoch_15']
  models = []
  for my_file in files:
    model = torch.load(my_file)
    model.use_cuda = False
    model.eval()
    models.append(model)
  _, _, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, _, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  src_vocab_size = len(src_vocab)
  trg_vocab_size = len(trg_vocab)

  nmt = EnsembleNMT(models)
  nmt.eval()

  if torch.cuda.is_available():
    nmt.cuda()
  else:
    nmt.cpu()

  with open('data/output_tanay.txt', 'w') as f_write:
    for ix in range(len(src_test)):
      test_src_batch = to_var(torch.unsqueeze(src_test[ix],1), volatile=True)
      test_trg_batch = to_var(torch.unsqueeze(trg_test[ix],1), volatile=True)
      test_src_batch = test_src_batch.view(-1, 1)
      test_trg_batch = test_trg_batch.view(-1, 1)
      
      sys_out_batch = nmt(test_src_batch, test_trg_batch)
      for j in range(sys_out_batch.size()[1]):
          sent = []
          for i in range(1, sys_out_batch.size()[0]):
              # print(sys_out_batch[i,j].data.numpy()[0])
              sent.append(trg_vocab.itos[sys_out_batch[i,j].data.numpy()[0]])
          print(' '.join(sent).encode('utf-8').strip())


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)

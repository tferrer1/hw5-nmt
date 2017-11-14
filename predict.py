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
# feel free to add more arguments as you need


def to_var(input, volatile=True):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main(options):

  _, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_test_src, batched_test_src_mask, sort_index = utils.tensor.advanced_batchize(src_test, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])

  src_vocab_size = len(src_vocab)
  trg_vocab_size = len(trg_vocab)

  nmt = NMT(src_vocab_size, trg_vocab_size)
  #nmt = torch.load(open('results/model.nll.epoch', 'rb'), pickle_module = "dill")
  nmt.eval()

  if torch.cuda.is_available():
    nmt.cuda()
  else:
    nmt.cpu()

  with open('data/output.txt', 'w') as f_write:
    for batch_i in utils.rand.srange(len(batched_test_src)):
      test_src_batch = to_var(batched_test_src[batch_i]) 
      batch_result = nmt(test_src_batch)
      s = ""
      for ix in batch_result:
        idx = np.argmax(ix.data.cpu().numpy())

        if idx == 2: # if <s>, don't write it
          continue
        if idx == 3: # if </s>, end the loop
          break
        
        s += trg_vocab.itos[idx] + " "

      print s.encode('utf-8')
      s += '\n'
      f_write.write(s.encode('utf-8'))  	


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)




  #  # main testing loop
  # last_test_avg_loss = float("inf")
  # for epoch_i in range(options.epochs):
  #   logging.info("At {0}-th epoch.".format(epoch_i))
  #   # srange generates a lazy sequence of shuffled range
  #   for i, batch_i in enumerate(utils.rand.srange(len(batched_test_src))):
  #     test_src_batch = to_var(batched_test_src[batch_i])  # of size (src_seq_len, batch_size)
  #     test_trg_batch = to_var(batched_test_trg[batch_i])  # of size (src_seq_len, batch_size)
  #     test_src_mask = to_var(batched_test_src_mask[batch_i])
  #     test_trg_mask = to_var(batched_test_trg_mask[batch_i])

  #     sys_out_batch = nmt(test_src_batch, test_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 
  #     test_trg_mask = test_trg_mask.view(-1)
  #     test_trg_batch = test_trg_batch.view(-1)
  #     test_trg_batch = test_trg_batch.masked_select(test_trg_mask)
  #     test_trg_mask = test_trg_mask.unsqueeze(1).expand(len(test_trg_mask), trg_vocab_size)
  #     sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
  #     sys_out_batch = sys_out_batch.masked_select(test_trg_mask).view(-1, trg_vocab_size)

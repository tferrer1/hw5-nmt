import torch
import torch.nn as nn
import dill
from torch.autograd import Variable

def to_var(input, volatile=False):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class ATTN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ATTN, self).__init__()
        self.Wi = nn.Linear(in_dim, in_dim, bias=False)
        self.Wo = nn.Linear(out_dim, in_dim, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
    def forward(self, input, h):
        semi = torch.unsqueeze(self.Wi(h), 0).expand_as(input)
        score = torch.t(self.softmax(torch.t(torch.sum(input * semi, dim=2))))
        score = torch.unsqueeze(score.contiguous(),2)
        s_tilde = torch.sum(score * input, dim=0)
        c_t = self.tanh(self.Wo(torch.cat([s_tilde, h], dim=1)))
        return c_t

class NMT(nn.Module):
    def __init__(self, trg_vocab_size):
        super(NMT, self).__init__()
        params = torch.load(open("data/model.param", 'rb'), pickle_module=dill)
        # encoding embedding
        self.EEMB = nn.Embedding(num_embeddings=36616, embedding_dim=300)
        self.EEMB.weight.data = params['encoder.embeddings.emb_luts.0.weight']
        # encoding
        self.ENC = nn.LSTM(input_size=300, hidden_size=512, bidirectional=True)
        self.ENC.weight_ih_l0.data = params['encoder.rnn.weight_ih_l0']
        self.ENC.weight_hh_l0.data = params['encoder.rnn.weight_hh_l0']
        self.ENC.bias_ih_l0.data =  params['encoder.rnn.bias_ih_l0']
        self.ENC.bias_hh_l0.data =  params['encoder.rnn.bias_hh_l0']
        self.ENC.weight_ih_l0_reverse.data = params['encoder.rnn.weight_ih_l0_reverse']
        self.ENC.weight_hh_l0_reverse.data = params['encoder.rnn.weight_hh_l0_reverse']
        self.ENC.bias_ih_l0_reverse.data =  params['encoder.rnn.bias_ih_l0_reverse']
        self.ENC.bias_hh_l0_reverse.data =  params['encoder.rnn.bias_hh_l0_reverse']
        # attention
        self.ATTN = ATTN(1024, 2048)
        self.ATTN.Wi.weight.data = params['decoder.attn.linear_in.weight']
        self.ATTN.Wo.weight.data = params['decoder.attn.linear_out.weight']
        # decoding
        self.DEC = nn.LSTM(input_size=1324, hidden_size=1024)
        self.DEC.weight_ih_l0.data = params['decoder.rnn.layers.0.weight_ih']
        self.DEC.weight_hh_l0.data = params['decoder.rnn.layers.0.weight_hh']
        self.DEC.bias_ih_l0.data = params['decoder.rnn.layers.0.bias_ih']
        self.DEC.bias_hh_l0.data = params['decoder.rnn.layers.0.bias_hh']
        # decoding embedding
        self.DEMB = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=300)
        self.DEMB.weight.data = params['decoder.embeddings.emb_luts.0.weight']
        # generator
        self.GEN = nn.Linear(in_features=1024, out_features=trg_vocab_size)
        self.GEN.weight.data =  params['0.weight']
        self.GEN.bias.data = params['0.bias']
        # miscellaneous
        self.logsoftmax = nn.LogSoftmax()
    
    def forward(self, input_src_batch, input_trg_batch=None):
        training = input_trg_batch != None
        
        if training:
            sent_len = input_trg_batch.size()[0]
        else:
            sent_len = input_src_batch.size()[0]

        encoder_input = self.EEMB(input_src_batch)
        encoder_output, (hidden, context) = self.ENC(encoder_input)

        seq_len = encoder_output.size()[0]
        batch_size = encoder_output.size()[1]

        hidden = hidden.permute(1,2,0).contiguous().view(batch_size, 1024)
        context = context.permute(1,2,0).contiguous().view(batch_size, 1024)

        if training:
            output = to_var(torch.zeros(sent_len, batch_size, 23262))
        else:
            output = to_var(torch.zeros(1, batch_size, 23262))
            word = to_var(torch.LongTensor(batch_size).fill_(2)) #

        for i in xrange(1, sent_len):
            c_t = self.ATTN(encoder_output, hidden)
            if training:
                decoder_input = torch.cat([c_t, self.DEMB(input_trg_batch[i-1])], dim=1)
            else:
                decoder_input = torch.cat([c_t, self.DEMB(word)], dim=1)

            decoder_input = decoder_input.unsqueeze(0)

            hidden = hidden.unsqueeze(0); context = context.unsqueeze(0)

            _, (hidden,context) = self.DEC(decoder_input, (hidden, context))

            hidden = hidden[0]; context = context[0]

            word = self.logsoftmax(self.GEN(hidden))

            if training:
                output[i] = word
            else:
                output = torch.cat([output, torch.unsqueeze(word, 0)], 0)
                _, word = torch.max(word, dim=1) 

        return output

import torch
import torch.nn as nn
import dill

params = torch.load(open("data/model.param", 'rb'), pickle_module=dill)

class ATTN(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(ATTN, self).__init__()
        self.Wi = torch.rand(dim_in, dim_in) 
        self.Wo = torch.rand(dim_in, dim_out)
  def forward(self, h, ht_1): # review these inputs
    score = []
    for s in range(len(h)):
        x = torch.mm(torch.transpose(h[s],1,2), torch.mm(self.Wi, ht_1))
        score.append(x)
    a=[]
    for s in range(len(h)):
        a.append(score[s]/ sum(score))
    s_tilde = 0
    for s in range(len(h)):
        s_tilde += a[s] * h[s]
    x = torch.cat((s_tilde, ht_1), 1) 
    c_t = nn.Tanh(torch.mm(x, self.Wo))
    return c_t

class NMT(nn.Module):
    def __init__(self):
        super(NMT, self).__init__()
        # encoding embedding
        self.EEM = nn.Embedding(num_embeddings=36616, embedding_dim=300)
        self.EEM.weight.data = params['encoder.embeddings.emb_luts.0.weight']
        # encoding
        self.ENC = nn.RNN(input_size=300, hidden_size=512, bidirectional=True)
        self.ENC.weight_ih_l0.data = params['encoder.rnn.weight_ih_l0']
        self.ENC.weight_hh_l0.data = params['encoder.rnn.weight_hh_l0']
        self.ENC.bias_ih_l0.data =  params['encoder.rnn.bias_ih_l0']
        self.ENC.bias_hh_l0.data =  params['encoder.rnn.bias_hh_l0']
        self.ENC.weight_ih_l0_reverse.data = params['encoder.rnn.weight_ih_l0_reverse']
        self.ENC.weight_hh_l0_reverse.data = params['encoder.rnn.weight_hh_l0_reverse']
        self.ENC.bias_ih_l0_reverse.data =  params['encoder.rnn.bias_ih_l0_reverse']
        self.ENC.bias_hh_l0_reverse.data =  params['encoder.rnn.bias_hh_l0_reverse']
        # attention
        self.ATT = ATTN(1024, 2048)
        self.ATT.Wi.data = params['decoder.attn.linear_in.weight'] # should we include .data here?
        self.ATT.Wo.data = params['decoder.attn.linear_out.weight'] # same question
        # decoding
        self.DEC = nn.RNN(input_size=1324, hidden_size=1024)
        self.DEC.weight_ih_l0.data =    params['decoder.rnn.layers.0.weight_ih']
        self.DEC.weight_hh_l0.data = params['decoder.rnn.layers.0.weight_hh']
        self.DEC.bias_ih_l0.data = params['decoder.rnn.layers.0.bias_ih']
        self.DEC.bias_hh_l0.data = params['decoder.rnn.layers.0.bias_hh']
        # generator
        self.GEN = nn.Linear(in_features=1024, out_features=23262)
        self.GEN.weight.data =  params['0.weight']
        self.GEN.bias.data = params['0.bias']
        # decoding embedding
        self.DEM = nn.Embedding(num_embeddings=23262, embedding_dim=300)
        self.DEM.weight.data = params['decoder.embeddings.emb_luts.0.weight']
    
    def forward(self, input):
        output = self.EEM(input)
        hidden1, output1 = self.ENC(output)
        context = self.ATT(output, hidden) #prolly wrong
        hidden2, output2 = self.DEC(context, OWE_t1)
        self.GEN(output2)
        
        

        return


"""input_ixs >>
    >> ENC WEM (red) >>
        >> tensor >>
          >> FSE (greenish) >>
            >> tensor + Hidden_state_t-1 (dark green) >>
                >> ATTN (yellow) >>
                    >> context tensor (lime green) + OWE_t-1 (magenta)>> 
    >> DECO LSTM >> Hidden_state_t
<< output_tensor (?) <<
  << GEN <<
<< tensor (orange) <<
  << DECO WEM <<
output emb (magenta) <<"""

# word embeddings
    # encoder.embeddings.emb_luts.0.weight torch.Size([src_vocab_size, src_word_emb_size] = [36616, 300]): the source word embedding
    # decoder.embeddings.emb_luts.0.weight torch.Size([trg_vocab_size, trg_word_emb_size] = [23262, 300]): the target word embedding

# forward source encoding:
   # encoder.rnn.weight_ih_l0 torch.Size([4 * encoder_hidden_size, src_word_emb_size] = [2048, 300]): the input connection to the gates of the LSTM, see here for how the weights are arranged
   # encoder.rnn.weight_hh_l0 torch.Size([4 * encoder_hidden_size, encoder_hidden_size] = [2048, 512]): the hidden connection to the gates of the LSTM, see here for how the weights are arranged
   # encoder.rnn.bias_ih_l0 torch.Size([4 * encoder_hidden_size] = [2048]): bias term for the input connections, same arrangement as above
   # encoder.rnn.bias_hh_l0 torch.Size([4 * encoder_hidden_size] = [2048]): bias term for the hidden connections, same arrangement as above

# backward source encoding (same thing as above): 
   # encoder.rnn.weight_ih_l0_reverse torch.Size([2048, 300])
   # encoder.rnn.weight_hh_l0_reverse torch.Size([2048, 512])
   # encoder.rnn.bias_ih_l0_reverse torch.Size([2048])
   # encoder.rnn.bias_hh_l0_reverse torch.Size([2048])

# decoder
# decoder.rnn.layers.0.weight_ih torch.Size([4 * decoder_hidden_size, trg_word_emb_size + context_vector_size] = [4096, 1324])
# decoder.rnn.layers.0.weight_hh torch.Size([4 * decoder_hidden_size, decoder_hidden_size] = [4096, 1024])
# decoder.rnn.layers.0.bias_ih torch.Size([4 * decoder_hidden_size] = [4096])
# decoder.rnn.layers.0.bias_hh torch.Size([4 * decoder_hidden_size] = [4096])

# generator
# 0.weight torch.Size([trg_vocab_size, decoder_hidden_size] = [23262, 1024])
# 0.bias torch.Size([decoder_hidden_size] = [1024])

# attention
    # decoder.attn.linear_in.weight torch.Size([1024, 1024]): Wi
    # decoder.attn.linear_out.weight torch.Size([1024, 2048]): Wo

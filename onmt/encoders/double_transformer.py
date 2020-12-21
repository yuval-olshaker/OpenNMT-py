"""
Implementation of double transformer
"""
import torch

from onmt.decoders import TransformerDecoder
from onmt.encoders import TransformerEncoder
from onmt.encoders.encoder import EncoderBase

from torch import Tensor
import torch.nn.functional as f


import torch.nn as nn

from onmt.encoders.encoder import EncoderBase



class DoubleTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, opt, embeddings, tg_embeddings=None):
        super(DoubleTransformerEncoder, self).__init__()
        self.first_encoder = TransformerEncoder.from_opt(opt, embeddings, 0)
        self.decoder = TransformerDecoder.from_opt(opt, tg_embeddings)
        self.second_encoder = TransformerEncoder.from_opt(opt, tg_embeddings)
        self.bptt = False
        self.counter = 0

    @classmethod
    def from_opt(cls, opt, embeddings, tg_embeddings=None):
        """Alternate constructor."""
        return cls(opt, embeddings, tg_embeddings)


    def forward(self, src, lengths=None, dec_in=None, bptt=False):
        """See :func:`EncoderBase.forward()`"""
        enc_state, memory_bank, lengths = self.first_encoder(src, lengths)
        if self.bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=False)


        weights = self.decoder.embeddings.word_lut.weight # we need to multiply by the embeddings to C

        # multiply by weights(t) - to vocab dimensions
        dec_out = torch.tensordot(dec_out, weights.t(), ([2], [0]))
        dec_out_temp = dec_out

        # gumbel softmax - choose the words we want from the vocab
        dec_out = nn.functional.gumbel_softmax(dec_out, tau=0.01, hard=True, dim=2)

        # multiply by weights back to embeddings dimensions
        dec_out = torch.tensordot(dec_out, weights, ([2], [0]))

        # check the embeddings every 3000 steps
        self.counter += 1
        if self.counter % 3000:
            temp = self.decoder.embeddings.do_first
            maxs = torch.argmax(dec_out_temp, dim=2)
            self.decoder.embeddings.do_first = True
            emb1 = self.decoder.embeddings(maxs.unsqueeze_(-1))
            self.decoder.embeddings.do_first = False
            emb2 = self.decoder.embeddings(dec_out)
            self.decoder.embeddings.do_first = temp
            with open('a.txt', 'a') as wr:
                wr.write('emb with argmax:\n')
                wr.write(emb1)
                wr.write('\n')
                wr.write('\n')
                wr.write('our emb:\n')
                wr.write(emb2)
                wr.write('\n')
                wr.write('\n')
                wr.write('\n')
                wr.write('\n')

        lengths2 = torch.tensor([dec_out.shape[0], dec_out.shape[1]]).to('cuda')
        enc_state2, memory_bank2, lengths2 = self.second_encoder(dec_out, lengths2)

        return enc_state2, memory_bank2, lengths2, dec_out

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)

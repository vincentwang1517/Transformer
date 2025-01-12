import torch
import torch.nn as nn

from model.model import *
from model.encoder import *
from model.decoder import *

########################################
''' Projection Layer '''
########################################
'''
A linear layer to project (..., d_model) to (..., vocab_size)
'''
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    

########################################
''' Transformer '''
########################################
class Transformer(nn.Module):
    
    def __init__(self, 
                 encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    '''
    src: src language
    '''
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
        
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

########################################
''' Builder '''
########################################

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
    """ Build transformer

    Args:
        src_seq_len (int): Sequence length with padded 
        tgt_seq_len (int): 
    """
    # Encoder
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    # Decoder
    tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize Parameters #[NOTE]
    for p in transformer.parameters():
        if p.dim() > 1: # weight only
            nn.init.xavier_uniform_(p)
            
    return transformer
    
    
import torch
import torch.nn as nn

from model.model import *

class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedFowardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])
        
        # ##
        # self.norm_1 = LayerNormalization()
        # self.norm_2 = LayerNormalization()
        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x
    
    # def forward2(self, x, mask):
    #     x_org = x
    #     x = self.norm_1(x)
    #     x = self.self_attention_block(x, x, x, mask)
    #     x = x_org + self.dropout_1(x)
        
    #     x_org = x
    #     x = self.norm_2(x)
    #     x = self.feed_forward_block(x)
    #     x = x_org + self.dropout_2(x)
        
    #     return x
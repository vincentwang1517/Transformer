import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

########################################
''' GET Dataset '''
########################################
def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    get_max_seq_len(ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # Build datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    valid_ds = BilingualDataset(valid_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False)
    
    return train_dl, valid_dl, tokenizer_src, tokenizer_tgt


########################################
''' Tokenizer '''
########################################
'''
[Description]: The tokenizer converts raw text into a sequence of tokens (numeric IDs)
'''
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_max_seq_len(ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang):
    src_max_len = 0
    tgt_max_len = 0
    for item in ds:
        src_seq = src_tokenizer.encode(item['translation'][src_lang]).ids
        tgt_seq = tgt_tokenizer.encode(item['translation'][tgt_lang]).ids
        src_max_len = max(src_max_len, len(src_seq))
        tgt_max_len = max(tgt_max_len, len(tgt_seq))
    print(f"Max source sequence length: {src_max_len}")
    print(f"Max target sequence length: {tgt_max_len}")


########################################
''' Dataset '''
########################################
def casual_mask(size):
    return torch.tril(torch.ones(1, size, size)).bool()

class BilingualDataset(Dataset):
    """
    What are important thinfs of a dataset?
    (1) The actual dat (__init__)
    (2) The length of the dataset (__len__)
    (3) How to retrieve an item (__getitem__). This should RETURN A TENSOR as input to the model
    """
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """Constructor

        Args:
            ds: dataset
            tokenizer_src: Source language tokenizer
            tokenizer_tgt: Target language tokenizer
            src_lang: Source launguage type
            tgt_lang: Target launguage type
            seq_len
        """
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # The special tokens ([SOS], [EOS], [PAD], [UNK]) will have the same IDs for both source (src) and target (tgt) tokenizers 
        # if they share the same vocabulary or are trained independently but in a similar way. 
        # This is because special tokens are explicitly defined and reserved during tokenizer training or initialization, 
        # and their IDs are consistent across tokenizers unless explicitly configured otherwise.
        self.sos_token = torch.tensor(tokenizer_src.encode("[SOS]").ids, dtype=torch.int64)
        self.eos_token = torch.tensor(tokenizer_src.encode("[EOS]").ids, dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_src.encode("[PAD]").ids, dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx: any) -> any:
        # text
        src_text = self.ds[idx]['translation'][self.src_lang]
        tgt_text = self.ds[idx]['translation'][self.tgt_lang]
        
        # tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # padding size
        enc_padding_size = self.seq_len - len(enc_input_tokens) - 2 # [SOS] and [EOS]
        dec_padding_size = self.seq_len - len(dec_input_tokens) - 1 # [SOS]
        if (enc_padding_size < 0): raise ValueError("Source input sequence > seq_len")
        if (dec_padding_size < 0): raise ValueError("Target input sequence > seq_len")
        
        # Concatenate inputs
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, 
                self.pad_token.repeat(enc_padding_size)   
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.pad_token.repeat(dec_padding_size)
            ]
        )
        label = torch.cat( # ground truth
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token, 
                self.pad_token.repeat(dec_padding_size)
            ]
        )
        
        assert encoder_input.shape[0] == self.seq_len and decoder_input.shape[0] == self.seq_len and label.shape[0] == self.seq_len
        
        return {
            "encoder_input": encoder_input, # (seg_len, )
            "decoder_input": decoder_input, # (seq_len, )   
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) # This is used in multi-head attention, whose shape is (batch, h, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.shape[0]), # (1, 1, seq_len)
            "label": label, 
            "src_text": src_text, 
            "tgt_text": tgt_text
        }
        
        
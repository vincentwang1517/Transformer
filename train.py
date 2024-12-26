import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from tqdm import tqdm # A fast, extensible progress bar for loops and other iterable objects

from model.transformer import *
from dataset import *
from config import *

def train_model(config):
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    '''
    To train a model. we need the following components:
    (1) dataset & data loader
    (2) model
    (3) loss function
    (4) optimizer
    '''
    
    # Load the dataset
    train_dl, valid_dl, tokenizer_src, tokenizer_tgt = get_ds(config)
    # model
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    model = build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model']).to(device)
    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    # Tensorboard
    writer = tb.SummaryWriter(config['experiment_name'])
    # others
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    
    # --------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- TRAINING ------------------------------------------------ #
    # --------------------------------------------------------------------------------------------------------- #
    for epoch in range(initial_epoch, config['num_epochs']):
        # ----- training ---------------------------------------------------------------------------------------------------- #
        model.train() # set the model to training mode 
        batch_iterator = tqdm(train_dl, desc=f"Epoch: {epoch: 02d}") # batch iterator with progress bar
        
        for batch in batch_iterator:
            # data
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len) --> Used on multi-head attention scores, whose dimension is (batch, h, seq_len, seq_len
            decoder_mask = batch["decoder_mask"].to(device) # (batch, 1, seq_len, seq_len)
            label = batch["label"].to(device) # (batch, seq_len)
            
            # forward
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)
            
            # calculate loss
            # CrossEntropyLoss expects the ground truth to be integer class indices, not probabilities
            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1)) # (batch * seq_len, tgt_vocab_size) vs. (batch * seq_len)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) # display additional custom information at the end of the progress bar
            
            # Log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()
            
            # Backpropagation
            loss.backward()
            
            # Update the weights (now gradient information has been passed to model.parameters())
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        # Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_filename)
        
        # ----- validation ---------------------------------------------------------------------------------------------------- #
        model.eval()
        batch_iterator = tqdm(valid_dl, desc=f"Validation: {epoch: 02d}")
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")
        for batch in batch_iterator:
            # encoder
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.shape[0] == 1, "Validation batch size must be 1"
            encoder_output = model.encode(encoder_input, encoder_mask)
            
            # Generate translation output using GREEDY DECODING
            decoder_input = torch.tensor([sos_idx], dtype=torch.int64).unsqueeze(0).to(device) # (1, 1)
            while True:
                if decoder_input.shape[1] >= config['seq_len']:
                    break
                # decoder output
                decoder_mask = casual_mask(decoder_input.shape[1]).to(device)
                decoder_output = model.decode(encoder_input, encoder_mask, decoder_input, decoder_mask) # (1, cur_len, d_model)
                # find token
                next_prob = model.project(decoder_output[:, -1]) 
                next_token = torch.argmax(next_prob, dim=-1, keepdim=False) # (1,)
                # concat decoder_input & token
                decoder_input = torch.cat([decoder_input, next_token], dim=-1)
                if next_token.item() == eos_idx:
                    break
                
            # ---------- Now, decoder input is your translation result ---------- #
            
            
                
                
            
            
            

        
            
        
if __name__ == '__main__':
    config = get_config()
    train_model(config)
        
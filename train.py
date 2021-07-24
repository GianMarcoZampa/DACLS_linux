import torch
import os
import torch.nn as nn
from torch.nn.functional import interpolate
import time
from pesq import pesq
from pystoi import stoi
from tqdm.notebook import tqdm
import pandas as pd
from model import Demucs
from dataset import Audio_dataset, Collate_fn
from loss import Multi_STFT_loss


def simulation_init():
    L = 5  # Number of layers
    H = 64 # Number of hidden channels
    K = 8  # Layer kernel size
    S = 2  # Layer stride
    U = 2  # Resampling factor

    demucs = Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, LSTM=True, bidirectional=True)
    optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    loss_func = Multi_STFT_loss()

    return demucs, optimizer, loss_func


def train_one_epoch(train_dataloader, device, model, optimizer, loss_func):
    model = model.to(device)
    model.train()
    epoch_mean_loss = 0.0

    for noisy_data, clean_data in tqdm(train_dataloader):

        # Send data to device: cuda or cpu
        noisy_data = noisy_data.to(device)
        clean_data = clean_data.to(device)

        # Resample at 16000
        noisy_data = interpolate(noisy_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)
        clean_data = interpolate(clean_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(noisy_data)

        # Compute loss
        loss = loss_func(outputs, clean_data)

        # Backward propagation
        loss.backward()

        # Optimization (update weights)
        optimizer.step()

        # Accumulate loss
        epoch_mean_loss += loss.item()

    epoch_mean_loss = epoch_mean_loss / len(train_dataloader)

    return epoch_mean_loss


def evaluate_one_epoch(test_dataloader, device, model):
    model = model.to(device)
    model.eval()
    epoch_mean_pesq = 0.0
    epoch_mean_stoi = 0.0

    for noisy_data, clean_data in tqdm(test_dataloader):
        
        # Resample at 16000
        noisy_data = interpolate(noisy_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)
        clean_data = interpolate(clean_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)

        # Evaluate model output
        cleaned_data = model.forward(noisy_data)

        # pesq() requires numpy format input
        clean_data = clean_data[0,0,:].numpy()
        cleaned_data = cleaned_data[0,0,:].detach().numpy()
        noisy_data = noisy_data[0,0,:].numpy()   

        pesq_val = pesq(16000, clean_data, cleaned_data, 'wb')

        stoi_val = stoi(clean_data, cleaned_data, 16000, extended=False)

        epoch_mean_pesq += pesq_val
        epoch_mean_stoi += stoi_val

    epoch_mean_pesq = epoch_mean_pesq / len(test_dataloader)
    epoch_mean_stoi = epoch_mean_stoi / len(test_dataloader)

    return epoch_mean_pesq, epoch_mean_stoi



def train(epochs, batch_size, saved_model, saved_loss, saved_pesq, saved_stoi, path=''):

    demucs, optimizer, loss_func = simulation_init()
    #print(f'Model - {demucs}, Optimizer - {optimizer}, Loss function - {loss_func}')
    #demucs.load_state_dict(torch.load(os.path.join(path, saved_model)))


    train_dataset = Audio_dataset(os.path.join(path, 'train_meta_file.csv'),
                                  os.path.join(path, 'dataset/train'))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   collate_fn=Collate_fn(),
                                                   shuffle=True,
                                                   num_workers=1)

    print(f'Train dataloader loaded - {len(train_dataloader)} batches in total')

    test_dataset = Audio_dataset(os.path.join(path, 'test_meta_file.csv'),
                                  os.path.join(path, 'dataset/test'))

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size,
                                                   collate_fn=Collate_fn(),
                                                   shuffle=True,
                                                   num_workers=1)
    

    print(f'Test dataloader loaded - {len(test_dataloader)} batches in total')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Start training with {device}')

    loss_list = []
    pesq_list = []
    stoi_list = []
    min_mean_loss = 1000

    # Training
    for epoch in range(1, epochs + 1):
        epoch_time = time.time()

        # Train one epoch
        epoch_mean_loss = train_one_epoch(train_dataloader, device, demucs, optimizer, loss_func)
        loss_list.append(epoch_mean_loss)

        # Evaluate one epoch
        epoch_mean_pesq, epoch_mean_stoi = evaluate_one_epoch(test_dataloader, device, demucs)
        pesq_list.append(epoch_mean_pesq)
        stoi_list.append(epoch_mean_stoi)

        # Print epoch result
        print(f'Epoch {epoch}/{epochs} - Loss: {epoch_mean_loss:.6f} - Pesq: {epoch_mean_pesq:.6f} - Stoi: {epoch_mean_stoi:.6f} - elapsed time: {(time.time() - epoch_time):.3f}')

        # Save the model if loss decreases
        if epoch_mean_loss < min_mean_loss:
            torch.save(demucs.state_dict(), os.path.join(path, saved_model))
            min_mean_loss = epoch_mean_loss
            print('Model checkpoint saved')
        
    # Save loss, pesq, stoi
    df = pd.DataFrame(loss_list) 
    df.to_csv(os.path.join(path, saved_loss))
    df = pd.DataFrame(pesq_list) 
    df.to_csv(os.path.join(path, saved_pesq))
    df = pd.DataFrame(stoi_list) 
    df.to_csv(os.path.join(path, saved_stoi))
            
    return loss_list


if __name__ == '__main__':
    lossl = train(1,                                               # epochs
                  1,                                               # batch size
                  saved_model = 'exp001_model.pt',                 # saved model file
                  saved_loss = 'exp001_loss.csv',                  # saved loss file
                  saved_pesq = 'exp001_pesq.csv',                  # saved pesq file
                  saved_stoi = 'exp001_stoi.csv',                  # saved stoi file
                  path = '')                                       # dataset metafile path

    print('Training done')
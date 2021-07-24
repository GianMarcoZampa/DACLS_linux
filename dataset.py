import torchaudio
import torch
from torch.nn.functional import interpolate
import pandas as pd
import os

class Audio_dataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, data_dir):
        """
        :param csv_file: csv file path containg meta data
        :param data_dir: data directory path
        """
        self.meta_file = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.sr = 16000
        
    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wav_file = self.meta_file.iloc[idx, 0]
        noisy_sample, noisy_sr = torchaudio.load(os.path.join(self.data_dir, 'noisy', wav_file))
        clean_sample, clean_sr = torchaudio.load(os.path.join(self.data_dir, 'clean', wav_file))

        noisy_stddev, noisy_mean = torch.std_mean(noisy_sample, dim=-1)
        clean_stddev, clean_mean = torch.std_mean(clean_sample, dim=-1)
        noisy_sample = (noisy_sample-noisy_mean.unsqueeze(1))/noisy_stddev.unsqueeze(1)
        clean_sample = (clean_sample-clean_mean.unsqueeze(1))/clean_stddev.unsqueeze(1)

        return noisy_sample, clean_sample


class Collate_fn():
    """
    A variant of collate_fn that pads according to the longest sequence in a batch of sequences
    """
    def pad_collate(self, batch):
        # find longest sequence inside batch
        max_len = max(map(lambda x: x[0].shape[1], batch))
        max_len = ((max_len//960)+1)*960 # zero-pad ad un multiplo intero di 20ms (960 campioni @ 48kHz)
        batch = list(map(lambda dims: (self.pad_tensor(dims[0], pad=max_len, dim=1), self.pad_tensor(dims[1], pad=max_len, dim=1)), batch))
        
        # stack all sequences
        noisy_sample_padded = torch.stack(tuple(map(lambda x: x[0], batch)), dim=0)
        clean_sample_padded = torch.stack(tuple(map(lambda x: x[1], batch)), dim=0)

        return noisy_sample_padded, clean_sample_padded

    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

    def __call__(self, batch):
        return self.pad_collate(batch)



def test(path):
    train_dataset = Audio_dataset(os.path.join(path, 'train_meta_file.csv'),
                                  os.path.join(path, 'dataset/train'))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   collate_fn=Collate_fn(),
                                                   shuffle=True,
                                                   num_workers=1)

    for noisy_data, clean_data in train_dataloader:

        noisy_data = interpolate(noisy_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)
        clean_data = interpolate(clean_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)

        print (noisy_data.shape, clean_data.shape)


if __name__ == '__main__':
    test('')
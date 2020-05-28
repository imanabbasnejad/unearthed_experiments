import torch
import ipdb
import h5py
import numpy as np


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(dataset_h5, self).__init__()

        self.file = h5py.File(in_file, 'r')
        self.n_f, self.nx, self.ny = self.file['features'].shape
        self.n_h,  self.n_hx, self.n_hy, self.n_hz= self.file['heatmap'].shape

    def __getitem__(self, index):
        heatmap = self.file['heatmap'][index,:, :, :]
        features = self.file['features'][index, :, :]
        return heatmap.astype('float32'), features.astype('float32')

    def __len__(self):
        return self.n_h


def DataLoader(h5_files_train, h5_files_test, batch_size):

    train_loader = torch.utils.data.DataLoader(dataset_h5(h5_files_train),
                                                    batch_size=batch_size, shuffle=False, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset_h5(h5_files_test), batch_size=200)

    return train_loader, validation_loader


def next_batch(dataloader_iterator):
    next_batch_tensor = dataloader_iterator.next()
    return next_batch_tensor[0], next_batch_tensor[1]

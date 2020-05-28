from netZoo import Net
from load_data_torch import DataLoader
from load_data_torch import next_batch
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import time


def get_train_names():
    h5_files = ['data_0.h5', 'data_1.h5', 'data_2.h5',
                'data_3.h5', 'data_4.h5', 'data_5.h5',
                'data_6.h5']

    return h5_files


def train(fine_tune=False, n_epoch=100):
    h5_files = get_train_names()
    net = Net()
    net.cuda()
    print(net)
    #
    if fine_tune:
        checkpoint = torch.load('saved_model')
        net.load_state_dict((checkpoint['state_dict']))
        net.train()
    criterion = nn.BCEWithLogitsLoss().cuda()
    for epoch in range(0, n_epoch):
        for h5_file in h5_files:

            train_loader, test_loader = DataLoader(h5_files_train='./h5_files/'+h5_file,
                                                   h5_files_test='./h5_files/data_7.h5',
                                                   batch_size=25)

            optimizer = optim.Adam(net.parameters(), lr=0.001,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=0,
                                   amsgrad=False)
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):

                heatmap, features_reshaped = data
                heatmap_reshaped = (heatmap.permute(0, 3, 1, 2)+1e-15)/(255. +1e-15)

                if True not in torch.isnan(features_reshaped):
                    outputs = net(features_reshaped.cuda())
                    optimizer.zero_grad()
                    loss = criterion(outputs.cuda(), heatmap_reshaped.cuda())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    if i % 20 == 0:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 20))
                        running_loss = 0.0

        acc_all = []

        for i, data in enumerate(test_loader, 0):
            heatmap, features_reshaped = data
            heatmap_reshaped = (heatmap.permute(0, 3, 1, 2) + 1e-15) / (255. + 1e-15)

            if True not in torch.isnan(features_reshaped):
                outputs = net(features_reshaped.cuda())
                pred = torch.max(outputs, 1)[1].cpu()
                pred_classes = pred.detach().numpy()

        print (np.mean(acc_all))

    state = {'epoch': epoch + 1,
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'losslogger': loss, }

    torch.save(state, './saved_model')

if __name__ == '__main__':
    train()
import ipdb;ipdb.set_trace()


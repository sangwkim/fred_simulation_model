from utils_010 import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pkbar
import os
import numpy as np

learning_rate = 1e-3
batch_size = 256
epochs = 400
log_interval = 10
shuffle = True
load_weight = True
test_model = False
slice_len = 1000

def train(log_interval, model, device, train_loader, optimizer, epoch, kbar, test_model):
    if not test_model:
        model.train()
    else:
        model.eval()
    N_count = 0
    train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view([X.size(0),-1])
        N_count += X.size(0)
        optimizer.zero_grad()
        pred = model(X)
        dist = torch.distributions.Normal(pred[:,:output_len*output_dim],pred[:,output_len*output_dim:])
        nlloss = -dist.log_prob(y).to(device)
        loss = torch.mean(nlloss)
        if not test_model:
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * X.size(0)
        kbar.update(batch_idx)
    train_loss /= N_count
    print('Epoch', epoch)
    print('\nTrain Set: Average loss: {:.6f}\n'.format(train_loss))
    return train_loss

def validate(model, device, optimizer, test_loader, kbar, test_model, save_folder, min_error):
    model.eval()
    N_count = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device).view([X.size(0),-1])
            N_count += X.size(0)
            pred = model(X)
            dist = torch.distributions.Normal(pred[:,:output_len*output_dim],pred[:,output_len*output_dim:])
            nlloss = -dist.log_prob(y).to(device)
            loss = torch.mean(nlloss)
            test_loss += loss.item() * X.size(0)
            kbar.update(batch_idx)
    test_loss /= N_count
    print('\nTest set: Average loss:{:.4f}'.format(test_loss))
    if test_loss < min_error and not test_model:
        min_error = test_loss
        torch.save(model.state_dict(), os.path.join(save_folder,'best_model.pth'))
        print("Epoch {} model saved!".format(epoch+1))
    return test_loss, min_error

if __name__ == '__main__':
    save_folder = './results/fredTwin_011/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    batch_size = 64
    validation_split = 0.2
    dataset = fred_dataset()
    dataset_size = len(dataset)//slice_len
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(40)
    if shuffle:
        np.random.shuffle(indices)
    train_indices = [ind*slice_len+ii for ind in indices[split:] for ii in range(slice_len)]
    val_indices = [ind*slice_len+ii for ind in indices[:split] for ii in range(slice_len)]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print('training data size: ', len(train_indices), 'validation data size: ', len(val_indices))
    
    params = {
        'batch_size': batch_size,
        'num_workers': 0,
        'pin_memory': True
    }

    train_loader = DataLoader(dataset, **params, sampler=train_sampler)
    valid_loader = DataLoader(dataset, **params, sampler=valid_sampler)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    FRED = fred_twin(args=[], input_channels=input_dim).to(device)

    if load_weight or test_model:
        FRED.load_state_dict(torch.load(save_folder+'best_model.pth'))
        epoch_train_losses = np.load(save_folder+'training_loss.npy').tolist()
        epoch_valid_losses = np.load(save_folder+'validation_loss.npy').tolist()
    else:
        epoch_train_losses = []
        epoch_valid_losses = []

    if torch.cuda.device_count() > 1:
    	print("Using", torch.cuda.device_count(), "GPUs")
    	FRED = nn.DataParallel(FRED)

    nn_params = FRED.parameters()
    optimizer = torch.optim.Adam(nn_params, lr=learning_rate)

    min_error = 1000
    train_per_epoch = len(train_indices)/batch_size
    valid_per_epoch = len(val_indices)/batch_size
    kbar_train = pkbar.Kbar(target=train_per_epoch, width=8)
    kbar_valid = pkbar.Kbar(target=valid_per_epoch, width=8)

    for epoch in range(epochs):
        train_loss = train(log_interval, FRED, device, train_loader, optimizer, epoch, kbar_train, test_model)
        valid_loss, min_error = validate(FRED, device, optimizer, valid_loader, kbar_valid, test_model, save_folder, min_error)
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        if not test_model:
            np.save(save_folder+'training_loss.npy', np.array(epoch_train_losses))
            np.save(save_folder+'validation_loss.npy', np.array(epoch_valid_losses))
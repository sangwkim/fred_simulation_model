import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

input_len = 50
input_dim = 5
output_len = 1
output_dim = 2

class fred_twin(nn.Module):
    def __init__(self, args, input_channels=input_dim):
        super(fred_twin, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_channels,hidden_size=64).double()
        self.lstm2 = nn.LSTM(input_size=64,hidden_size=64).double()
        self.lstm3 = nn.LSTM(input_size=64,hidden_size=32).double()
        self.linear = nn.Linear(in_features=32,out_features=2*output_len*output_dim).double()
        self.softp = nn.Softplus()

    def forward(self, x):
        x = x.permute((1,0,2))
        out_lstm1, (h1, c1) = self.lstm1(x.double())
        out_lstm2, (h2, c2) = self.lstm2(out_lstm1)
        out_lstm3, (h3, c3) = self.lstm3(out_lstm2)
        out_linear = self.linear(torch.squeeze(h3,0))
        out_softp = self.softp(out_linear[:,output_len*output_dim:])
        out_last = torch.cat((out_linear[:,:output_len*output_dim],out_softp), 1)
        return out_last
    
class fred_dataset():
    def __init__(self):
        ds_01 = sio.loadmat('./data/rawdata.mat')['rawdata']
        ds_02 = sio.loadmat('./data/rawdata_1w.mat')['rawdata_1w']
        ds_03 = sio.loadmat('./data/rawdata_25w.mat')['rawdata_25w']
        ds_04 = sio.loadmat('./data/rawdata_ln.mat')['rawdata_ln']
        ds_05 = sio.loadmat('./data/rawdata_nw.mat')['rawdata_nw']
        xy = np.vstack((ds_01,ds_02,ds_03,ds_04,ds_05))
        self.x_data = torch.from_numpy(np.array(xy[:,[1,2,3,4,5]], dtype=np.float))
        self.y_data = torch.from_numpy(np.array(xy[:,[2,3]], dtype=np.float))
        idxs_sub_begin = np.where(xy[:,1]==0)[0]
        idxs_sub_begin = np.append(idxs_sub_begin,len(xy))
        self.idxs = []
        for ii in range(len(idxs_sub_begin)-1):
            self.idxs = np.append(self.idxs, np.arange(idxs_sub_begin[ii]+input_len, int(idxs_sub_begin[ii+1])-output_len-1))
        self.idxs = np.array(self.idxs, dtype=np.int)
        self.x_mean = torch.mean(self.x_data[self.idxs,:], axis=0)
        self.x_std = torch.std(self.x_data[self.idxs,:], axis=0)
        self.y_mean = torch.mean(self.y_data[self.idxs,:], axis=0)
        self.y_std = torch.std(self.y_data[self.idxs,:], axis=0)
        self.x_data = (self.x_data - self.x_mean)/self.x_std
        self.y_data = (self.y_data - self.y_mean)/self.y_std
        self.len = len(self.idxs)
    def __getitem__(self, index):
        return self.x_data[self.idxs[index]-input_len:self.idxs[index]], self.y_data[self.idxs[index]:self.idxs[index]+output_len]
    def __len__(self):
        return self.len

def load_model(save_folder, model):
    model.load_state_dict(torch.load(save_folder+'best_model.pth', map_location=torch.device('cpu')))

def rolling_stat(seq, stat='avg', winlen=50):
    pred = []
    seq = np.array(seq)
    for i in range(len(seq)):
        rol = seq[max(0,i-winlen//2):min(len(seq),i+winlen-winlen//2)]
        rol = rol[rol!=rol.max()]
        rol = rol[rol!=rol.min()]
        if stat == 'avg':
            r_stat = np.mean(rol)
        elif stat == 'std':
            r_stat = np.std(rol)
        else: raise
        pred.append(r_stat)
    return np.array(pred)

def show_ex6(model, dataset, device, state_num, val_indices):
    for i in range(0,len(val_indices),1000):
        idx_show = dataset.idxs[val_indices[i]]
        show_ex5(model, dataset, device, state_num, idx_show, show_len=500)

def show_ex5(model, dataset, device, state_num, idx_show=137750, show_len=1000, seed=13):
    model.eval()
    rolling_len = 50
    np.random.seed(seed)
    pred = []
    pred_mean = []
    pred_std = []
    inp = dataset.x_data[idx_show:idx_show+50,:].unsqueeze(0).clone()
    g_tru = dataset.y_data[idx_show:idx_show+show_len+50] * dataset.y_std + dataset.y_mean
    g_tru = g_tru.detach().numpy()
    for ii in range(show_len):
        outp = model(inp.to(device))
        outp = outp.to('cpu')
        outp_mean = outp[:,:2*output_len] * dataset.y_std + dataset.y_mean
        outp_mean = outp_mean.detach().numpy().squeeze(0)
        outp_std = outp[:,2*output_len:] * dataset.y_std
        outp_std = outp_std.detach().numpy().squeeze(0)
        pred_out = np.random.normal(outp_mean,outp_std)
        pred.append(pred_out)
        pred_mean.append(outp_mean)
        pred_std.append(outp_std)
        inp_new = dataset.x_data[idx_show+ii+51,:].unsqueeze(0).unsqueeze(0).clone()
        inp_new[0][0][[1,2]] = (torch.tensor(pred_out) - dataset.y_mean)/dataset.y_std
        inp = torch.cat((inp[:,1:,:].cpu(),inp_new),1)
    pred, pred_mean, pred_std = np.array(pred), np.array(pred_mean), np.array(pred_std)
    plt.figure()
    plt.plot(np.arange(50,show_len+50)/4., rolling_stat(pred[:,state_num],'avg',rolling_len), c='r')
    plt.plot(np.arange(0,show_len+50)/4., rolling_stat(g_tru[:,state_num],'avg',rolling_len), c='b')
    plt.fill_between(np.arange(50,show_len+50)/4., rolling_stat(pred[:,state_num],'avg',rolling_len)-2*rolling_stat(pred[:,state_num],'std',rolling_len), rolling_stat(pred[:,state_num],'avg',rolling_len)+2*rolling_stat(pred[:,state_num],'std',rolling_len), alpha=.3, fc='r')
    plt.fill_between(np.arange(0,show_len+50)/4., rolling_stat(g_tru[:,state_num],'avg',rolling_len)-2*rolling_stat(g_tru[:,state_num],'std',rolling_len), rolling_stat(g_tru[:,state_num],'avg',rolling_len)+2*rolling_stat(g_tru[:,state_num],'std',rolling_len), alpha=0.3, fc='b')
    plt.plot(np.arange(0,show_len+50)/4., g_tru[:,state_num], alpha=.3, c='b')
    plt.plot(np.arange(50,show_len+50)/4., pred[:,state_num], alpha=.3, c='r')
    plt.legend(['simulated','measured'])
    plt.xlabel('time (s)')
    if state_num == 0:
        plt.ylabel("diameter (e-5 m)")
    elif state_num == 1:
        plt.ylabel("spool speed")
    plt.show()
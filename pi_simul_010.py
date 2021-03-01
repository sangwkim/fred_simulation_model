from utils_010 import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

save_folder = './results/fredTwin_011/'

def simulate_pi(input_len, ref_traj, ext_f=10):

    global xm, xs

    xm, xs = dataset.x_mean.detach(), dataset.x_std.detach()
    
    err_diam_intgl, err_vel_intgl = 0, 0
    
    pred, act = [], []
    sa = torch.from_numpy(np.tile([0,0,0,0,0],(1,input_len,1))).to(device)
    for i in range(len(ref_traj)):
        if i == 0:
            action = np.array([0,0])
            act.append(action)
        sa[0,-1,[3,4]] = torch.tensor(action).to(device)
        with torch.no_grad():
            outp = FRED(sa)
        pred_out = torch.normal(outp[0,:2],outp[0,2:])
        pred.append(pred_out)
        sa_new = torch.zeros(sa[:,[0],:].size()).to(device)
        sa_new[0,0,[1,2]] = pred_out.float()
        sa_new[0,0,0] = sa[0,-1,0] + (sa[0,-1,4]*xs[4]+xm[4])/2600/xs[0] + 1/130/xs[0]
        sa = torch.cat((sa[:,1:,:],sa_new.double()),1)

        ext_speed = 9.45*2*np.pi*ext_f/200/16
        model_vel = 2 * ext_speed * 7.112**2 / (float(ref_traj[i])/1000)**2 / 20 / 2 / np.pi
        
        gp = 0.00015
        gi = 0.000015
        #gp = 2.73603343e-05
        #gi = 1.90500445e-06
    
        err_diam = 10*(pred_out[0]*xs[1]+xm[1]) - ref_traj[i]
        err_diam_intgl = err_diam_intgl + err_diam
        pi = gp * err_diam + gi * err_diam_intgl
        com_vel = model_vel + pi
    
        vp = 35
        vi = 20
        err_vel = float(12.5*(pred_out[1]*xs[2]+xm[2]))/840 - com_vel
        err_vel_intgl = err_vel_intgl + err_vel
        vpi = - vp * err_vel - vi * err_vel_intgl
        a0 = np.clip(vpi.cpu(),20,256)
        a0 = np.interp(a0,[20,256],[0,100])
        xx = np.linspace(0,100,400)
        yy = 4.221e-8*xx**5-5.874e-6*xx**4+2.514e-4*xx**3-5.58e-4*xx**2+0.1951*xx
        a0 = np.interp(a0,yy,xx)
        a1 = np.interp(ext_f,[5,30],[0,100])
        
        action = np.array([(a0-xm[3])/xs[3],(a1-xm[4])/xs[4]])
        act.append(action)
        
    act = np.stack(act[:-1])
    pred = torch.stack(pred).cpu().detach().numpy()
    
    return pred, act

if __name__ == '__main__':
    
    s_points = [450,450,550,350,550,400,500,350,500,400,550,350,500,400,550]
    trajectory = np.hstack([np.tile(s_point,200) for s_point in s_points])
    
    dataset = fred_dataset()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    FRED = fred_twin(args=[], input_channels=input_dim).to(device)
    if use_cuda: FRED.load_state_dict(torch.load(save_folder+'best_model.pth'))
    else: FRED.load_state_dict(torch.load(save_folder+'best_model.pth', map_location=torch.device('cpu')))

    if torch.cuda.device_count() > 0:
    	print("Using", torch.cuda.device_count(), "GPUs")
    	FRED = nn.DataParallel(FRED)
        
    pred, act = simulate_pi(input_len, trajectory, ext_f=10)
    
    np.savetxt('pi_sim_xx.txt',10*(pred[:,0]*xs[1].detach().numpy()+xm[1].detach().numpy()))
    
    plt.plot(10*(pred[:,0]*xs[1].detach().numpy()+xm[1].detach().numpy()))
    plt.plot(trajectory)

import argparse
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ODE demo Morris-Lecar')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4') #RCL modified default
parser.add_argument('--data_size', type=int, default=801) #RCL modified default
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--nspikes', type=int, default=2) #RCL added new option. Number of spikes sought in fit


args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print('Running in adjoint mode (CAM)')
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Notation from http://icwww.epfl.ch/~gerstner/SPNM/node21.html
# Values from http://www.math.pitt.edu/~bard/bardware/meth3/node18.html
feed = {
    # 'g1': 4.4,
    'g1': 4.0,
    'g2': 8.0,
    'gL': 2.0,

    'u1': -1.2,
    'u2': 18.0,
    # 'u3': 2.0,
    # 'u4': 30.0,
    'u3': 12.0,
    'u4': 17.4,

    'V1': 120.0,
    'V2': -84.0,
    'VL': -60.0,

    # 'tau_w': 1/0.04,
    'tau_w': 1 / (1 / 15.0),

    #'I': 40 #This creates 2 spikes
    'I': 42 #This would create 3 spikes, all else the same (including t below)
}

if args.nspikes==2:
    feed['I']=40 # 2 spikes
if args.nspikes==3:
    feed['I']=42 # 3 spikes

#RCL
init=[-30.0,0]
true_y0=torch.tensor([init],dtype=torch.float)
t = torch.linspace(0., 80., args.data_size)

class Lambda(nn.Module):

    def forward(self, t, y):
        #RCL
        x=y.t()
        dx = torch.zeros_like(x)

        #The states are
        #V - x[0]
        #w - x[1]

        m = 1.0/(1+torch.exp(2*(feed['u1']-x[0])/feed['u2']))
        wst = 1.0/(1+torch.exp(2*(feed['u3']-x[0])/feed['u4']))

        tau = feed['tau_w']/torch.cosh((x[0]-feed['u3'])/(2*feed['u4']))

        dx[0] = -feed['g1']*m*(x[0]-feed['V1']) -feed['g2']*x[1]*(x[0]-feed['V2']) -feed['gL']*(x[0]-feed['VL']) + feed['I']
        dx[1] = (wst-x[1])/tau

        return dx.t()

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method=args.method)

#RCL Set an explicit seed for noise for reproducibility
torch.manual_seed(0)
#Add noise to observations
noise_dist = torch.distributions.normal.Normal(0, 1.0)
noise_shape = torch.Size([true_y.size()[0], true_y.size()[1], true_y.size()[2]])
noise_samples = noise_dist.sample(sample_shape=noise_shape)
true_y = true_y + noise_samples

# Plot membrane potential only
def plot_membranepotential(t, outs, outs_pred, outs_init):
    '''
    outs is a torch tensor of shape torch.Size([num_steps+1, num_comps])
    '''

    # Extract states from tensor
    # Make a copy of x! Because the detached numpy object points to the same torch data.
    x = torch.t(outs)[0].detach().numpy().copy()  # https://discuss.pytorch.org/t/cant-call-numpy-on-variable-that-requires-grad/20763
    t = t.numpy()

    if outs_pred is not None:
        x_pred = torch.t(outs_pred)[0].detach().numpy()
        x_init = torch.t(outs_init)[0].detach().numpy()

    f, axarr = plt.subplots(1, 1, figsize=(6, 4))
    # axarr.set_title(r'time series')
    # axarr.plot(t,x,'-.')
    axarr.plot(t, x)
    axarr.set_ylabel(r'$u(t)$ (mV)')
    axarr.set_xlabel(r'$t$ (ms)')
    if outs_pred is not None:
        # axarr.plot(t,x_pred,'-')
        axarr.plot(t, x_pred)
        axarr.plot(t, x_init)
        axarr.legend(['u(t) ground truth','u(t) fit','u(t) initial'])
    else:
        pass

#For three spikes
#Seed: 250 Loss: 4.970378875732422 MAPE: 0.011531185110410055 Iter: 42 Init: {'g1': 3.7718234062194824, 'g2': 7.543927192687988, 'gL': 3.637662172317505, 'tau_w': 17.372488021850586} Best: {'g1': 3.995425224304199, 'g2': 8.130254745483398, 'gL': 1.9915188550949097, 'tau_w': 15.3668794631958}
if args.nspikes==3:
    Init={'g1': 3.7718234062194824, 'g2': 7.543927192687988, 'gL': 3.637662172317505, 'tau_w': 17.372488021850586}

#For two spikes
#Seed: 120 Loss: 25.15618896484375 MAPE: 0.06687008539835612 Iter: 191 Init: {'g1': 4.142364501953125, 'g2': 13.31101131439209, 'gL': 6.274686336517334, 'tau_w': 14.54284954071045} Best: {'g1': 4.249972820281982, 'g2': 8.661396026611328, 'gL': 2.033829927444458, 'tau_w': 16.580965042114258}
if args.nspikes==2:
    Init={'g1': 4.142364501953125, 'g2': 13.31101131439209, 'gL': 6.274686336517334, 'tau_w': 14.54284954071045}

for k, v in Init.items():
    feed[k]=v

with torch.no_grad():
    init_y = odeint(Lambda(), true_y0, t, method=args.method)

# For three spikes
# Seed: 250 Loss: 4.970378875732422 MAPE: 0.011531185110410055 Iter: 42 Init: {'g1': 3.7718234062194824, 'g2': 7.543927192687988, 'gL': 3.637662172317505, 'tau_w': 17.372488021850586} Best: {'g1': 3.995425224304199, 'g2': 8.130254745483398, 'gL': 1.9915188550949097, 'tau_w': 15.3668794631958}
if args.nspikes==3:
    Best={'g1': 3.995425224304199, 'g2': 8.130254745483398, 'gL': 1.9915188550949097, 'tau_w': 15.3668794631958}

# For two spikes
# Seed: 120 Loss: 25.15618896484375 MAPE: 0.06687008539835612 Iter: 191 Init: {'g1': 4.142364501953125, 'g2': 13.31101131439209, 'gL': 6.274686336517334, 'tau_w': 14.54284954071045} Best: {'g1': 4.249972820281982, 'g2': 8.661396026611328, 'gL': 2.033829927444458, 'tau_w': 16.580965042114258}
if args.nspikes==2:
    Best={'g1': 4.249972820281982, 'g2': 8.661396026611328, 'gL': 2.033829927444458, 'tau_w': 16.580965042114258}

for k, v in Best.items():
    feed[k] = v

with torch.no_grad():
    best_y = odeint(Lambda(), true_y0, t, method=args.method)

plot_membranepotential(t, true_y.squeeze(),best_y.squeeze(),init_y.squeeze())

plt.show()

#if args.nspikes==3:
#    plt.savefig('MorrisLecar_ThreeSpikeFit_Traces.eps')
#if args.nspikes==2:
#    plt.savefig('MorrisLecar_TwoSpikeFit_Traces.eps')
#plt.clf()
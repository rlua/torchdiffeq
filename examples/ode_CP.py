import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=201) #RCL modified default
parser.add_argument('--batch_time', type=int, default=20) #RCL modified default
parser.add_argument('--batch_size', type=int, default=10) #RCL modified default
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=10) #RCL modified default
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--npendulums', type=int, default=3) #RCL added new option
args = parser.parse_args()

#RCL
N_=args.npendulums
N2_=2*N_
print('Number of pendulums and state variables:', N_, N2_)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#RCL
init = [0.0]*N2_
for i in range(N_):
    if i % 10 == 0:  # This was the default initial condition
        init[2 * i] = 0.5
true_y0=torch.tensor([init])
t = torch.linspace(0., 10., args.data_size)

#Provide ground-truth parameter values
    #>>> import scipy.stats
    #>>> x=scipy.stats.truncnorm.rvs(-3,3,size=100)
    #>>> x+(2*3.14159*0.5)**2
k_groundtruth = [  8.92867921,   9.71355317,  10.17317032,   9.77653158,
     9.78911964,   9.2805172 ,  10.44233473,   9.14408993,
     9.44462192,  11.30223152,   8.3543937 ,  11.20229649,
     9.93619751,   9.50667967,   7.80333545,   9.56068608,
    10.6651111 ,   9.06818809,   9.64369855,  10.79708007,
    10.60891094,   9.46320946,   8.62527907,   7.73927375,
     9.03999325,  11.78582019,  10.72829545,  10.75744361,
     9.44664905,   9.8590231 ,   8.68748667,   9.45238485,
    10.7371234 ,  10.00651714,   9.8417173 ,   9.39353523,
     8.34719479,  10.19805438,  10.09884707,  10.34284743,
    11.70144574,   9.60323213,   8.19781305,   9.78805522,
     9.6723787 ,  10.15046324,  10.09365032,   9.82512569,
     8.22736749,   9.40463433,   9.70438509,  10.30899965,
    10.16364906,  10.1533394 ,  11.94416848,   9.2644704 ,
    10.61335796,   9.60804568,   8.13432541,  10.12324133,
    10.17164954,   8.31119946,   9.88018304,  10.41149407,
     9.54476624,   9.00306368,  10.86919719,  10.29095393,
     9.96165922,   8.67444141,   8.64311475,   9.52429671,
     9.74441168,   8.36718697,   9.5196447 ,  10.97737869,
     9.06881692,  10.61413509,  10.64773855,  10.12200223,
    11.15692462,   7.98755224,  10.12438577,   9.32193699,
     9.35883653,   9.34403633,   8.93980613,  10.63222048,
    11.15481569,  10.56336653,  12.22789202,   8.70155198,
     8.41090053,   9.19563812,  11.88613871,   9.80109468,
     9.28963387,   9.04412782,   8.11297509,  10.57329466]

feed={'rho':4.0}
for i in range(N_):
    var_name='k%d' % (i+1)
    feed[var_name]=k_groundtruth[i]

class Lambda(nn.Module):

    def forward(self, t, y):
        #RCL
        x=y.t()
        dx = torch.zeros_like(x)
        for i in range(N_):
            ipos=2*i #index into x of coordinate
            ivel=ipos+1 #index into x of velocity
            iposnext=(ipos+2)%N2_
            iposprev=(ipos-2)%N2_
            dx[ipos]=x[ivel]
            dx[ivel]=-feed['k%d'%(i+1)]*torch.sin(x[ipos])-feed['rho']*(2*x[ipos]-x[iposnext]-x[iposprev])

        return dx.t()

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_phase = fig.add_subplot(122, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,v')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        #RCL Modified limits for CP
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('v')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        #RCL Modified limits for CP
        ax_phase.set_xlim(-0.5, 0.5)
        ax_phase.set_ylim(-2, 2)

        #RCL Commented out vector field plotting
        # ax_vecfield.cla()
        # ax_vecfield.set_title('Learned Vector Field')
        # ax_vecfield.set_xlabel('x')
        # ax_vecfield.set_ylabel('y')
        #
        # #RCL Modified limits
        # y, x = np.mgrid[0:4:21j, 0:4:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)
        #
        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # #RCL Modified limits
        # ax_vecfield.set_xlim(0, 4)
        # ax_vecfield.set_ylim(0, 4)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.01)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        #RCL
        self.feed={}
        #RCL Initial parameter values set to half the ground truth (from global feed)
        for k, v in feed.items():
            self.feed[k]=torch.nn.Parameter(0.5*torch.tensor(v))
            self.register_parameter(k,self.feed[k])
        #for v in self.feed.values():
        #    v.requires_grad_()

    def forward(self, t, y):
        #RCL
        #y may contain multiple batches, take the transpose
        x=y.t()
        dx = torch.zeros_like(x)
        for i in range(N_):
            ipos=2*i #index into x of coordinate
            ivel=ipos+1 #index into x of velocity
            iposnext=(ipos+2)%N2_
            iposprev=(ipos-2)%N2_
            dx[ipos]=x[ivel]
            dx[ivel]=-self.feed['k%d'%(i+1)]*torch.sin(x[ipos])-self.feed['rho']*(2*x[ipos]-x[iposnext]-x[iposprev])

        return dx.t()

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc()
    # RCL
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1.0e-0)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    #RCL Use MSE instead of abs
    lossfunc = torch.nn.MSELoss()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0.squeeze(), batch_t) #RCL had to call squeeze
        #print(pred_y, batch_y)
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #RCL Modified loss function, use MSE instead of abs
        loss = lossfunc(pred_y,batch_y.squeeze()) #RCL Had to call squeeze
        #print(func.feed,pred_y.requires_grad)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                #RCL Use MSE instead of abs
                loss=lossfunc(pred_y,true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                #RCL
                #print(func.feed)
                ii += 1

        end = time.time()

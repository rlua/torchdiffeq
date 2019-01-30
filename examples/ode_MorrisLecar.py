import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4') #RCL modified default
parser.add_argument('--data_size', type=int, default=801) #RCL modified default
parser.add_argument('--batch_time', type=int, default=40) #RCL modified default
parser.add_argument('--batch_size', type=int, default=20) #RCL modified default
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=10) #RCL modified default
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
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

    'I': 40}

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

        #tau = self.params['tau_w']/np.cosh((x[0]-self.params['u3'])/(2*self.params['u4']))
        tau = feed['tau_w']/torch.cosh((x[0]-feed['u3'])/(2*feed['u4']))

        dx[0] = -feed['g1']*m*(x[0]-feed['V1']) -feed['g2']*x[1]*(x[0]-feed['V2']) -feed['gL']*(x[0]-feed['VL']) + feed['I']
        dx[1] = (wst-x[1])/tau

        return dx.t()

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method=args.method)

#Add noise to observations
noise_dist = torch.distributions.normal.Normal(0, 1.0)
noise_shape = torch.Size([true_y.size()[0], true_y.size()[1], true_y.size()[2]])
noise_samples = noise_dist.sample(sample_shape=noise_shape)
true_y = true_y + noise_samples

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

#RCL When states are partially observed, only initial values of observed states can be used
def get_batch_partialObs(idx, pred_y):
    '''
    :param idx: indices of observed states
    :param pred_y: simulated state trajectory using current parameter estimates
    :return:
    '''
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time), args.batch_size, replace=False))
    #Use initial conditions from simulated trajectory. This avoids using any information from ground truth (including unobserved states)
    batch_y0 = pred_y[s]  # (M, 1, D)
    #print(batch_y0)
    #batch_y0[:,:,idx] = true_y[s][:,:,idx]
    #print(batch_y0)
    batch_t = t[:args.batch_time]  # (T)
    #RCL The unobserved states from ground truth at the first batch timepoint must be replaced also.
    #RCL On second thought, it is OK to leave the unobserved states in true_y, as these won't be included when
    #computing the loss
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax_traj = fig.add_subplot(111, frameon=False) #111 one row, one column, first panel
    #ax_phase = fig.add_subplot(122, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('u (mV, membrane potential)')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'r')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], 'r--')
        ax_traj.set_xlim(t.min(), t.max())
        #RCL Modified limits
        ax_traj.set_ylim(-70, 70)
        ax_traj.legend()

        # ax_phase.cla()
        # ax_phase.set_title('Phase Portrait')
        # ax_phase.set_xlabel('x')
        # ax_phase.set_ylabel('v')
        # ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        # ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        # #RCL Modified limits
        # ax_phase.set_xlim(-0.5, 0.5)
        # ax_phase.set_ylim(-2, 2)

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
        #plt.pause(1)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        #RCL
        self.feed={}
        for k, v in feed.items():
            if k in ['tau_w']:
            #if k in ['g1','g2','gL','tau_w']:
            #if k in ['g1', 'g2', 'gL', 'tau_w','V1','V2','VL']:
                self.feed[k]=torch.nn.Parameter(1.5*torch.tensor(v,dtype=torch.float))
                self.register_parameter(k,self.feed[k])
            else:
                self.feed[k] = torch.tensor(v,dtype=torch.float)


    #Make sure to use self.feed in this forward method
    def forward(self, t, y):
        #RCL
        #y may contain multiple batches, take the transpose
        x=y.t()
        dx = torch.zeros_like(x)

        m = 1.0/(1+torch.exp(2*(self.feed['u1']-x[0])/self.feed['u2']))
        wst = 1.0/(1+torch.exp(2*(self.feed['u3']-x[0])/self.feed['u4']))

        tau = self.feed['tau_w']/torch.cosh((x[0]-self.feed['u3'])/(2*self.feed['u4']))

        dx[0] = -self.feed['g1']*m*(x[0]-self.feed['V1']) - self.feed['g2']*x[1]*(x[0]-self.feed['V2']) -self.feed['gL']*(x[0]-self.feed['VL']) + self.feed['I']
        dx[1] = (wst-x[1])/tau

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
    #RCL
    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1.5e-1)
    #optimizer = optim.LBFGS(func.parameters(), lr=2e-1)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    #RCL Use MSE instead of abs
    lossfunc = torch.nn.MSELoss()

    #RCL
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=100,verbose=True)

    for itr in range(1, args.niters + 1):
        with torch.no_grad():
            pred_y_notbatch = odeint(func, true_y0, t, method=args.method)
            #batch_y0, batch_t, batch_y = get_batch()
            batch_y0, batch_t, batch_y = get_batch_partialObs([0],pred_y_notbatch)
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0.squeeze(), batch_t, method=args.method) #RCL had to call squeeze
        #print(pred_y, batch_y)
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #RCL Modified loss function, use MSE instead of abs
        #Use membrane potential only, slice and get 0-component of state
        loss = lossfunc(pred_y[:,:,0],batch_y.squeeze()[:,:,0]) #RCL Had to call squeeze
        #print(func.feed,pred_y.requires_grad)
        loss.backward()

        #RCL Experiment with LBFGS
        #https: // pytorch.org / docs / stable / optim.html
        def closure():
            optimizer.zero_grad()
            #output = model(input)
            pred_y_ = odeint(func, batch_y0.squeeze(), batch_t, method=args.method)
            #loss = loss_fn(output, target)
            loss_ = lossfunc(pred_y_[:,:,0], batch_y.squeeze()[:,:,0])
            loss_.backward()
            return loss_

        optimizer.step() #Use for Adam
        #optimizer.step(closure) #Use for LBFGS

        #RCL
        #scheduler.step(loss)

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                #RCL Use MSE instead of abs
                loss=lossfunc(pred_y,true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                #RCL
                #for param in func.parameters():
                #    print(param.grad)
                #for param_group in optimizer.param_groups:
                #    print(param_group['lr'])
                ii += 1

        end = time.time()

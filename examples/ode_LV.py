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
parser.add_argument('--niters', type=int, default=100) #RCL modified default
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true') #RCL modified default
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#RCL
true_y0 = torch.tensor([[1., 0.5]])
t = torch.linspace(0., 10., args.data_size)
feed = {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0, 'delta': 1.0}


class Lambda(nn.Module):

    def forward(self, t, y):
        #RCL Lotka-Volterra modification
        x=y.t()
        dx = torch.zeros_like(x)
        dx[0] = feed['alpha']*x[0] - feed['beta']*x[0]*x[1]
        dx[1] = -feed['gamma']*x[1] + feed['delta']*x[0]*x[1]
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
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        #RCL Modified limits for Lotka Volterra
        ax_traj.set_ylim(0, 4)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        #RCL Modified limits for Lotka Volterra
        ax_phase.set_xlim(0, 4)
        ax_phase.set_ylim(0, 4)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        #RCL Modified limits for Lotka Volterra
        y, x = np.mgrid[0:4:21j, 0:4:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        #RCL Modified limits for Lotka Volterra
        ax_vecfield.set_xlim(0, 4)
        ax_vecfield.set_ylim(0, 4)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.01)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        #RCL Lotka-Volterra modification
        #Set initial parameter values here
        #self.feed={'alpha': 0.5*torch.tensor(feed['alpha']),
        #     'beta': torch.tensor(feed['beta']),
        #     'gamma': 0.5*torch.tensor(feed['gamma']),
        #     'delta': torch.tensor(feed['delta'])}
        #self.feed['alpha'].requires_grad_()
        #self.feed['gamma'].requires_grad_()

        #RCL Initial parameter values set to half the ground truth (from global feed)
        self.feed = {'alpha': torch.nn.Parameter(0.5*torch.tensor(feed['alpha'])),
                     'beta': (torch.tensor(feed['beta'])),
                     'gamma': torch.nn.Parameter(0.5*torch.tensor(feed['gamma'])),
                     'delta': (torch.tensor(feed['delta']))}

        #https://discuss.pytorch.org/t/nn-parameters-vs-nn-module-register-parameter/18641/2
        self.register_parameter('alpha',self.feed['alpha'])
        self.register_parameter('gamma',self.feed['gamma'])


    def forward(self, t, y):
        #RCL Lotka-Volterra modification
        #y may contain multiple batches, take the transpose
        x=y.t()
        dx = torch.zeros_like(x)
        dx[0] = self.feed['alpha']*x[0] - self.feed['beta']*x[0]*x[1]
        dx[1] = -self.feed['gamma']*x[1] + self.feed['delta']*x[0]*x[1]
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
    #RCL Lotka-Volterra modification
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    #optimizer = optim.RMSprop([func.feed['alpha'],func.feed['gamma']], lr=1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1e-1) #This one works too
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    #RCL Use MSE instead of abs
    lossfunc = torch.nn.MSELoss()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0.squeeze(), batch_t) #RCL had to call squeeze
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #RCL Modified loss function
        loss = lossfunc(pred_y,batch_y.squeeze()) #RCL Had to call squeeze
        #print(loss)
        loss.backward()
        optimizer.step()
        #RCL
        #print(func.feed)

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                loss=lossfunc(pred_y,true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                #RCL
                #print(func.feed)
                ii += 1

        end = time.time()

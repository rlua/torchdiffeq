import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4')
parser.add_argument('--data_size', type=int, default=801) #RCL modified default
parser.add_argument('--batch_time', type=int, default=40) #RCL modified default
parser.add_argument('--batch_size', type=int, default=20) #RCL modified default
parser.add_argument('--niters', type=int, default=1000) #RCL modified default
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--test_freq2', type=int, default=10) #RCL Added new option
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--nhidden', type=int, default=50) #RCL Added new option
parser.add_argument('--wscale', type=float, default=100) #RCL Added new option
#parser.add_argument('--lbfgs', action='store_true')
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

    #'I': 40 #This creates 2 spikes
    #'I': 42 #This would create 3 spikes, all else the same (including t below)
    'I':60 #4 spikes
}

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
    #true_y[:,:, 1] = true_y[:,:, 1]*true_y[:,:, 0].max()/true_y[:,:, 1].max() #Scale the w-state to be of the same magnitude as V
    true_y[:, :, 1] = true_y[:, :, 1]*args.wscale

#RCL Set an explicit seed for noise for reproducibility
torch.manual_seed(0)
#Add noise to observations
noise_dist = torch.distributions.normal.Normal(0, 1.0)
noise_shape = torch.Size([true_y.size()[0], true_y.size()[1], true_y.size()[2]])
noise_samples = noise_dist.sample(sample_shape=noise_shape)
#true_y = true_y + noise_samples

np.random.seed(0) #RCL
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
    batch_y0[:,:,idx] = true_y[s][:,:,idx]
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
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr ,viz=args.viz):

    if viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-70, 70)
        #ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-70, 70)
        ax_phase.set_ylim(0, 70)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[0:70:21j, -70:70:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-70, 70)
        ax_vecfield.set_ylim(0, 70)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        print('Number of nodes in a hidden layer:', args.nhidden)
        self.net = nn.Sequential(
            #nn.Linear(2, 2),
            nn.Linear(2, args.nhidden),
            #nn.Tanh(),
            nn.ReLU(),

            nn.Linear(args.nhidden, args.nhidden),
            nn.ReLU(),

            #nn.Linear(args.nhidden, args.nhidden),
            #nn.ReLU(),

            nn.Linear(args.nhidden, 2),
            #nn.Linear(2, 2),
        )

        torch.manual_seed(0) #RCL
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                print(m.weight.shape)
                #if m.weight.shape==(2,2):
                #    m.weight[0,1].requires_grad=False
                #    m.weight[1,0].requires_grad=False

        #self.scale_in1=torch.nn.Parameter(torch.tensor(1.0))
        #self.scale_in2 = torch.nn.Parameter(torch.tensor(1.0))
        #self.scale_out1=torch.nn.Parameter(torch.tensor(1.0))
        #self.scale_out2 = torch.nn.Parameter(torch.tensor(1.0))

        #self.register_parameter('scale_in1', self.scale_in1)
        #self.register_parameter('scale_in2', self.scale_in2)
        #self.register_parameter('scale_out1', self.scale_out1)
        #self.register_parameter('scale_out2', self.scale_out2)

    def forward(self, t, y):
        return self.net(y)

        # RCL modified
        #x=y.transpose(-1,0)
        #scale_in = torch.zeros_like(x)
        #scale_in[0] = self.scale_in1*x[0]
        #scale_in[1] = self.scale_in2*x[1]
        #return self.net(scale_in.transpose(-1,0))



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
    SavedNNModel = ODEFunc()
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    lossfunc = torch.nn.MSELoss()

    min_totalloss=1e10

    for itr in range(1, args.niters + 1):
        with torch.no_grad():
            pred_y_notbatch = odeint(func, true_y0, t, method=args.method)
            #batch_y0, batch_t, batch_y = get_batch()
            batch_y0, batch_t, batch_y = get_batch_partialObs([0,1],pred_y_notbatch)
        optimizer.zero_grad()
        #batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=args.method)
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #loss = torch.mean(torch.abs(pred_y[:,:,0] - batch_y[:,:,0])) #RCL Train on membrane potential
        loss = lossfunc(pred_y[:,:,0],batch_y[:,:,0]) #RCL Train on membrane potential
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
        #if itr == 1608:
        #if itr==1978:
        #if itr==980:
        #if itr==756:
        #if itr==731:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                loss = lossfunc(pred_y[:, :, 0], true_y[:, :, 0])  # RCL Train on membrane potential
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii, viz=False)
                ii += 1
                if loss.item() < min_totalloss:
                    min_totalloss=loss.item()
                    min_iter=itr
                    SavedNNModel.load_state_dict(func.state_dict())
            #RCL temporary
            #break

        end = time.time()

    print('Minimum total loss {:.6f} at iter {:d}'.format(min_totalloss,min_iter))


#Part two of fitting
# Export the NN model
# Fit the true model to the NN, by evaluating the MSE between NN-forward and TrueModel-forward at V,w values
# found in the ground truth
# Pitfalls:
#  1. No noise in data yet
#  2. Unobserved w-state is being used to train the NN (not in the loss function, but in initializing the minibatches

class TrueODEFunc(nn.Module):

    def __init__(self):
        super(TrueODEFunc, self).__init__()

        #RCL
        self.feed={}
        for k, v in feed.items():
            if k in ['tau_w']:
            #if k in ['g1','g2','gL','tau_w']:
            #if k in ['g1', 'g2', 'gL', 'tau_w','V1','V2','VL']:
                self.feed[k]=torch.nn.Parameter(1.0*torch.tensor(v,dtype=torch.float))
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

        dx[0] = -self.feed['g1']*m*(x[0]-self.feed['V1']) - self.feed['g2']*(x[1]/args.wscale)*(x[0]-self.feed['V2']) -self.feed['gL']*(x[0]-self.feed['VL']) + self.feed['I']
        dx[1] = args.wscale*(wst-(x[1]/args.wscale))/tau

        return dx.t()


if __name__ == '__main__':

    TrueModel = TrueODEFunc()
    #Freeze the parameters of the NN. We are going to fit the true model's forward function to the NN's forward
    for p in SavedNNModel.parameters():
        p.requires_grad = False
    #print(TrueModel.forward(t, true_y0))

    ii = 0

    #RCL
    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    #optimizer = optim.Adam(func.parameters(), lr=1.5e-1)
    optimizer = optim.Adam(TrueModel.parameters(), lr=1.5e-1)
    end = time.time()

    #RCL Use MSE instead of abs
    lossfunc = torch.nn.MSELoss()

    #RCL
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=100,verbose=True)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        NNf=torch.zeros([t.size()[0],2])
        Truef=torch.zeros([t.size()[0],2])
        for i in range(t.size()[0]):
            #print(NNf[i].shape,func.forward(0,true_y[i,0].view(1,-1)).shape,TrueModel.forward(0,true_y[i,0].view(1,-1)).shape)
            NNf[i]=SavedNNModel.forward(0,true_y[i,0].view(1,-1))
            Truef[i]=TrueModel.forward(0, true_y[i, 0].view(1,-1))
        loss = lossfunc(Truef,NNf)
        loss.backward()

        optimizer.step() #Use for Adam

        #RCL
        #scheduler.step(loss)

        if itr % args.test_freq2 == 0:
            with torch.no_grad():
                pred_y1 = odeint(SavedNNModel, true_y0, t, method=args.method)
                pred_y2 = odeint(TrueModel, true_y0, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                #RCL Use MSE instead of abs
                loss=lossfunc(pred_y1,pred_y2)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(pred_y1, pred_y2, TrueModel, ii, viz=True)
                ii += 1

        end = time.time()

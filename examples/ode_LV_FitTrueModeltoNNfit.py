import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['rk4','dopri5', 'adams'], default='rk4')
parser.add_argument('--data_size', type=int, default=201) #RCL modified default
parser.add_argument('--batch_time', type=int, default=20) #RCL modified default
parser.add_argument('--batch_size', type=int, default=10) #RCL modified default
parser.add_argument('--niters', type=int, default=1000) #RCL modified default
parser.add_argument('--test_freq', type=int, default=10) #RCL modified default
parser.add_argument('--test_freq2', type=int, default=10) #RCL Added new option
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--nhidden', type=int, default=50) #RCL Added new option
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#RCL
true_y0 = torch.tensor([[1., 0.5]])
t = torch.linspace(0., 10., args.data_size)
t_extend = torch.linspace(0., 20., 2*args.data_size)
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
    true_y = odeint(Lambda(), true_y0, t, method=args.method)
    true_y_extend = odeint(Lambda(), true_y0, t_extend, method=args.method)

#RCL Set an explicit seed for noise for reproducibility
torch.manual_seed(0)
#Add noise to observations
noise_dist = torch.distributions.normal.Normal(0, 0.2)
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

#Original
def visualize(true_y, pred_y, odefunc, itr, viz=args.viz, filename=None):

    if viz:

        # makedirs('png')
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12, 4), facecolor='white')
        # ax_traj = fig.add_subplot(131, frameon=False)
        # ax_phase = fig.add_subplot(132, frameon=False)
        # ax_vecfield = fig.add_subplot(133, frameon=False)
        # plt.show(block=False)

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(0, 4)
        #ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(0, 4)
        ax_phase.set_ylim(0, 4)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[0:4:21j, 0:4:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(0, 4)
        ax_vecfield.set_ylim(0, 4)

        fig.tight_layout()
        if filename is None:
            plt.savefig('png/{:03d}'.format(itr))
        else:
            plt.savefig('png/{}'.format(filename))

        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        print('Number of nodes in a hidden layer:',args.nhidden)
        self.net = nn.Sequential(
            nn.Linear(2, args.nhidden),
            #nn.Tanh(),
            nn.ReLU(), #RCL One layer nhidden 50,40,30,20,10 works. To a lesser extent 8,6,4 is close or getting there. 2 is not.
            #nn.Linear(args.nhidden, args.nhidden),
            #nn.Tanh(),
            nn.Linear(args.nhidden, 2),
        )

        torch.manual_seed(0) #RCL
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y) #RCL modified


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
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    lossfunc = torch.nn.MSELoss()

    min_totalloss=1e10

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=args.method)
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        loss = lossfunc(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                loss = lossfunc(pred_y, true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                #visualize(true_y, pred_y, func, ii, viz=False)
                ii += 1
                if loss.item() < min_totalloss:
                    min_totalloss=loss.item()
                    min_iter=itr
                    SavedNNModel.load_state_dict(func.state_dict())
                    visualize(true_y, pred_y, func, ii, viz=True, filename='LV_Best_NNfit.png')
                if itr==1:
                    visualize(true_y, pred_y, func, ii, viz=True, filename='LV_Start_NNfit.png')

        end = time.time()


#Part two of fitting
print('Begin fitting f_TrueModel to f_NN')

class TrueODEFunc(nn.Module):

    def __init__(self):
        super(TrueODEFunc, self).__init__()

        #RCL Initial parameter values set to half the ground truth (from global feed)
        self.feed = {'alpha': torch.nn.Parameter(0.0*torch.tensor(feed['alpha'])),
                     'beta': (torch.tensor(feed['beta'])),
                     'gamma': torch.nn.Parameter(0.0*torch.tensor(feed['gamma'])),
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


#https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
def plot_surface_3d_demo(model1,model2):
    '''
    ======================
    3D surface (color map)
    ======================

    Demonstrates plotting a 3D surface colored with the coolwarm color map.
    The surface is made opaque by using antialiased=False.

    Also demonstrates using the LinearLocator and custom formatting for the
    z axis tick labels.
    '''

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, 4, 0.25)
    Y = np.arange(0, 4, 0.25)
    X, Y = np.meshgrid(X, Y)

    #R = np.sqrt(X ** 2 + Y ** 2)
    #Z = np.sin(R.numpy())
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z1[i,j]=model1.forward(0,torch.tensor([[X[i,j],Y[i,j]]],dtype=torch.float))[0,0].numpy()
                Z2[i,j]=model2.forward(0,torch.tensor([[X[i,j],Y[i,j]]],dtype=torch.float))[0,0].numpy()
                #Z1[i,j]=model1.forward(0,torch.tensor([[X[i,j],Y[i,j]]],dtype=torch.float))[0,1].numpy()
                #Z2[i,j]=model2.forward(0,torch.tensor([[X[i,j],Y[i,j]]],dtype=torch.float))[0,1].numpy()

    # Plot the surface.
    surf1 = ax.plot_surface(X, Y, Z1, color='red', #cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(X, Y, Z2, color='blue', #cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(Z1.min(), Z1.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf1, shrink=0.5, aspect=5)
    #fig.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':

    ii = 0

    TrueModel = TrueODEFunc()
    #RCL Lotka-Volterra modification
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    #optimizer = optim.RMSprop([func.feed['alpha'],func.feed['gamma']], lr=1e-2)
    optimizer = optim.Adam(TrueModel.parameters(), lr=1e-1) #This one works too
    end = time.time()

    #RCL Use MSE instead of abs
    #lossfunc = torch.nn.MSELoss(reduction='elementwise_mean') #pytorch version 0.4.1, elementwise_mean is the default
    lossfunc = torch.nn.MSELoss()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        NNf=torch.zeros([t.size()[0],2])
        Truef=torch.zeros([t.size()[0],2])
        for i in range(t.size()[0]):
            #print(NNf[i].shape,func.forward(0,true_y[i,0].view(1,-1)).shape,TrueModel.forward(0,true_y[i,0].view(1,-1)).shape)
            NNf[i]=SavedNNModel.forward(0,true_y[i,0].view(1,-1))
            Truef[i]=TrueModel.forward(0, true_y[i,0].view(1,-1))
        loss = lossfunc(Truef,NNf)
        loss.backward()
        optimizer.step()

        if itr % args.test_freq2 == 0:
            with torch.no_grad():
                pred_y1 = odeint(SavedNNModel, true_y0, t, method=args.method)
                pred_y2 = odeint(TrueModel, true_y0, t, method=args.method)
                # loss = torch.mean(torch.abs(pred_y - true_y))
                # RCL Use MSE instead of abs
                loss = lossfunc(pred_y1, pred_y2)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if itr==1:
                    visualize(pred_y1, pred_y2, TrueModel, ii, viz=True, filename='LV_Start_TrueModelFit.png')
                    print(TrueModel.feed)
                    #plot_surface_3d_demo(SavedNNModel, TrueModel)
                elif itr==args.niters:
                    visualize(pred_y1, pred_y2, TrueModel, ii, viz=True, filename='LV_Final_TrueModelFit.png')
                    print(TrueModel.feed)
                    plot_surface_3d_demo(SavedNNModel, TrueModel)
                ii += 1

        end = time.time()


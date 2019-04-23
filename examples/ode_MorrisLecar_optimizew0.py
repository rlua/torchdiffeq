import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='ODE demo Morris-Lecar')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4') #RCL modified default
parser.add_argument('--data_size', type=int, default=801) #RCL modified default
parser.add_argument('--batch_time', type=int, default=40) #RCL modified default
parser.add_argument('--batch_size', type=int, default=20) #RCL modified default
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=1) #RCL modified default
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--lbfgs', action='store_true') #RCL added new option
parser.add_argument('--randomseed_initialparams', type=int, default=0) #RCL added new option
parser.add_argument('--nspikes', type=int, default=3) #RCL added new option. Number of spikes sought in fit


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
    #'tau_w': 2, #Thin spikes

    'I': 40 #This creates 2 spikes
    #'I': 42 #This would create 3 spikes, all else the same (including t below)
}

#RCL
init=[-30.0,0]
true_y0=torch.tensor([init],dtype=torch.float)
t = torch.linspace(0., 80., args.data_size)

def true_fromfile(filename):
    data=open(filename,'r').readlines()
    t=[]
    V=[]
    for line in data:
        cols=line.strip().split()
        if len(cols)<2:
            continue
        t.append(float(cols[0]))
        V.append([float(cols[1]),0])
    return len(t), torch.tensor(t), torch.tensor(V).unsqueeze(1)

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
#true_y[:,:,0] = (true_y + noise_samples)[:,:,0] #Add noise only to the membrane potential

#Transform data
#This convolution based approach is incomplete
def ConvGauss(x,length,sigma):
    #norm = tf.contrib.distributions.Normal(loc=tf.floor(length / 2), scale=sigma)
    norm = torch.distributions.normal.Normal(loc=length/2, scale=sigma)
    filter=[]
    for i in range(length):
        #print(torch.exp(norm.log_prob(i)))
        filter.append(torch.exp(norm.log_prob(i)))
    filter=torch.tensor([[filter]])
    print(filter)
    print(x.shape)
    return torch.nn.functional.conv1d(x[:,:,0].t().unsqueeze(0),filter,padding=(7,))

#print(ConvGauss(true_y,16,4))
#convgauss_true_y_V=ConvGauss(true_y,16,4)
#true_y[:,:,0]=convgauss_true_y_V.transpose(2,0)

#Second version
def SmoothSpikes(x,length,sigma):
    '''

    :param x: expect shape of yrue_y [data_size,0,2]
    :param length:
    :param sigma:
    :return:
    '''
    norm = torch.distributions.normal.Normal(loc=length/2, scale=sigma)
    filter=[]
    for i in range(length):
        filter.append(torch.exp(norm.log_prob(i)).item())
    sumprob=np.sum(filter)
    for i in range(length):
        filter[i]/=sumprob
    print(filter)
    for i in range(x.shape[0]):
        s = []
        for j in range(length):
            h=int(length/2)
            if i-h+j>=0 and i-h+j<x.shape[0]:
                s.append(x[i-h+j,0,0]*filter[j])
        if len(s)>0:
            x[i,0,0]=torch.tensor(np.mean(s),dtype=torch.float)
    true_y[:,:,0]=100*x[:,:,0]
#SmoothSpikes(true_y,200,10)

#######

#Load Fabrizio's time series for V(t)
#args.data_size, t, true_y = true_fromfile('ml_test3_rl.txt')
#print(args.data_size)
#print(t)
#print(true_y)

#Experiment to get w_guess(t) by integrating dw/dt and using V_obs(t)
#Assume lots of time-series data for V_obs(t), equally spaced.
def w_guess_Euler(V_obs, tau_w, u3, u4, w0, dt):
    w_tmp = torch.zeros_like(V_obs) #1D array
    w_tmp[0] = w0

    for i in range(1,V_obs.shape[0]):
        #simple Euler
        wst = 1.0 / (1 + torch.exp((u3 - V_obs[i]) / u4))
        tau = tau_w / torch.cosh((V_obs[i] - u3) / (2 * u4))
        w_tmp[i] = w_tmp[i-1] + (wst-w_tmp[i-1])/tau * dt

    return w_tmp

def w_guess(V_obs, tau_w, u3, u4, w0, dt):
    w_tmp = torch.zeros_like(V_obs) #1D array

    #w0 = 1.0 / (1 + torch.exp((u3 - V_obs[0]) / u4)) #wst

    w_tmp[0] = w0

    #RK2
    for i in range(1,V_obs.shape[0]):
        wst = 1.0 / (1 + torch.exp((u3 - V_obs[i-1]) / u4))
        tau = tau_w / torch.cosh((V_obs[i-1] - u3) / (2 * u4))
        V_av_k1 = (V_obs[i]+V_obs[i-1])/2
        w_k1 = (wst-w_tmp[i-1])/tau * dt
        wst = 1.0 / (1 + torch.exp((u3 - V_av_k1) / u4))
        tau = tau_w / torch.cosh((V_av_k1 - u3) / (2 * u4))
        k2 = (wst-w_tmp[i-1]-w_k1/2)/tau * dt
        w_tmp[i] = w_tmp[i - 1] + k2

    return w_tmp

def w_guess_RK4(V_obs, tau_w, u3, u4, w0, dt):
    w_tmp = torch.zeros_like(V_obs) #1D array

    #w0 = 1.0 / (1 + torch.exp((u3 - V_obs[0]) / u4)) #wst

    w_tmp[0] = w0

    def wst_func(V):
        return 1.0 / (1 + torch.exp((u3 - V) / u4))
    def tau_func(V):
        return tau_w / torch.cosh((V - u3) / (2 * u4))

    #RK4 TODO Check that V is used correctly in RK4 method
    for i in range(1,V_obs.shape[0]):
        V_av = (V_obs[i]+V_obs[i-1])/2.0
        w_k1 = (wst_func(V_obs[i-1])-w_tmp[i-1])/tau_func(V_obs[i-1])
        w_k2 = (wst_func(V_av)-w_tmp[i-1]-w_k1/2.0)/tau_func(V_av)
        w_k3 = (wst_func(V_av)-w_tmp[i-1]-w_k2/2.0)/tau_func(V_av)
        #w_k4 = (wst_func(V_av) - w_tmp[i - 1] - w_k3) / tau_func(V_av)
        w_k4 = (wst_func(V_obs[i]) - w_tmp[i - 1] - w_k3) / tau_func(V_obs[i])
        w_tmp[i] = w_tmp[i - 1] + 1/6.0*dt*(w_k1+2*(w_k2+w_k3)+w_k4)

    return w_tmp

def w0_guess(V_obs,u1,u2,g1,g2,gL,V1,V2,VL,I,dt):
    m = 1.0 / (1 + torch.exp(2 * (u1 - V_obs[0]) / u2))
    dVdt = (V_obs[2]-V_obs[0])/(2*dt)
    #Estimate w0 by solving for it in the equation at t=0:
    #dVdt = -g1 * m * (V_obs[0] - V1) - g2 * w0 * (V_obs[0] - V2) - gL * (V_obs[0] - VL) + I
    return (- dVdt -g1 * m * (V_obs[0] - V1) - gL * (V_obs[0] - VL) + I)/(g2 * (V_obs[0] - V2))


dt=t[1]-t[0]
print('dt:',dt)

#Augment membrane potential with simulated recovery variable w(t)
w0 = w0_guess(true_y.squeeze().t()[0],
              feed['u1'], feed['u2'],
              feed['g1'], feed['g2'], feed['gL'],
              feed['V1'], feed['V2'], feed['VL'], feed['I'], dt)
w_g_RK4 = w_guess_RK4(true_y.squeeze().t()[0], feed['tau_w'], feed['u3'], feed['u4'], w0, dt)
true_y[:,:,1]=w_g_RK4.unsqueeze(1)
true_y0=true_y[0]
#Write to file
torch.save(true_y, 'FG_ml_test3_rl_SimulateW.pt')


if False:
    import matplotlib.pyplot as plt

    #Visualizing how robust simulated w(t) from random parameters compared to ground truth
    w_g_Euler = w_guess_Euler(true_y.squeeze().t()[0], feed['tau_w'], feed['u3'], feed['u4'], 0, dt)
    w_g = w_guess(true_y.squeeze().t()[0], feed['tau_w'], feed['u3'], feed['u4'], 0, dt)
    w_g_RK4 = w_guess_RK4(true_y.squeeze().t()[0], feed['tau_w'], feed['u3'], feed['u4'], 0, dt)
    #plt.plot(t.numpy(),w_g_Euler.numpy())
    #plt.plot(t.numpy(),w_g.numpy())
    #plt.plot(t.numpy(), w_g_RK4.numpy())

    np.random.seed(0)

    plt.xlabel('t')
    plt.ylabel('w')
    legend=['Ground Truth']
    plt.plot(t.numpy(), true_y.squeeze().t()[1].numpy())
    for i in range(10):
        tau_w=np.random.uniform(0,2*feed['tau_w'])
        u3=np.random.uniform(0,2*feed['u3'])
        u4=np.random.uniform(0,2*feed['u4'])
        w0 = w0_guess(true_y.squeeze().t()[0],
                      feed['u1'], feed['u2'],
                      feed['g1'], feed['g2'], feed['gL'],
                      feed['V1'], feed['V2'], feed['VL'], feed['I'], dt)
        print(w0)
        #w0=0 #np.random.uniform(0,0.3)
        w_g = w_guess(true_y.squeeze().t()[0], tau_w, u3, u4, w0, dt)
        if True: #if i==-1:
            plt.plot(t.numpy(),w_g.numpy())
        legend.append('Random {:d}'.format(i+1))
    plt.legend(legend)
    #plt.legend(['Simulate Euler','Simulate RK2','Simulate RK4','Ground Truth'])
    plt.ylim(0,1.5)
    plt.show()
#######

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

#RCL When states are partially observed, only initial values of observed states can be used
def get_batch_partialObs(guess_w):
    '''
    :param idx: indices of observed states
    :param pred_y: simulated state trajectory using current parameter estimates
    :return:
    '''
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, 1, D)

    #batch_y0[:, :, 0] = true_y[s][:,:,0]
    batch_y0[:, :, 1] = guess_w[s].unsqueeze(1)

    #Write w_guess to true_y
    true_y[:,:,1]=guess_w.unsqueeze(1)

    #print(batch_y0)
    batch_t = t[:args.batch_time]  # (T)
    #RCL The unobserved states from ground truth at the first batch timepoint must be replaced also.
    #RCL On second thought, it is OK to leave the unobserved states in true_y, as these won't be included when
    #computing the loss
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 1, D)
    return batch_y0, batch_t, batch_y

#RCL When states are partially observed, only initial values of observed states can be used
def get_batch_partialObs_older(idx, pred_y, guess_w):
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

    batch_y0[:, :, 0] = true_y[s][:,:,0]
    batch_y0[:, :, 1] = guess_w[s].unsqueeze(1)

    #Write w_guess to true_y
    true_y[:,:,1]=guess_w.unsqueeze(1)

    #print(batch_y0)
    batch_t = t[:args.batch_time]  # (T)
    #RCL The unobserved states from ground truth at the first batch timepoint must be replaced also.
    #RCL On second thought, it is OK to leave the unobserved states in true_y, as these won't be included when
    #computing the loss
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 1, D)
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


def visualize(true_y, pred_y, odefunc, itr, plotV=True):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('u (mV, membrane potential)')
        if plotV:
            ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'r')
            ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], 'r--')
            ax_traj.plot(t.numpy(), 100*pred_y.numpy()[:, 0, 1], 'b--')
            ax_traj.set_ylim(-70, 70)
        else:
            ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 1], 'r')
            ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 1], 'r--')
            #ax_traj.plot(t.numpy(), pred_y.numpy(), 'r--') #when pred_y is w_g
            ax_traj.set_ylim(-1, 1)
        ax_traj.set_xlim(t.min(), t.max())
        #RCL Modified limits
        #ax_traj.set_ylim(-70, 70)
        #ax_traj.legend()

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

    def __init__(self, randomseed_initialparams):
        super(ODEFunc, self).__init__()

        #RCL
        np.random.seed(randomseed_initialparams)
        init_values_rand = {'g1': np.random.uniform(3.6,4.4),
                        'g2': np.random.uniform(1,25),
                        'gL': np.random.uniform(0.1,10),
                        'tau_w': np.random.uniform(5,25),

                        'u3': np.random.uniform(0, 2 * feed['u3']),
                        'u4' : np.random.uniform(0, 2 * feed['u4']),
                        }

        #tau_w = np.random.uniform(0, 2 * feed['tau_w'])
        #w0 = np.random.uniform(0, 0.3)

        # init_values_rand = {'g1': np.random.uniform(1,10),
        #                'g2': np.random.uniform(1,25),
        #                'gL': np.random.uniform(0.1,10),
        #                'tau_w': np.random.uniform(1,30),
        #
        #                 'u3' : np.random.uniform(0, 2 * feed['u3']),
        #                 'u4' : np.random.uniform(0, 2 * feed['u4']),
        #
        #                 'u1': np.random.uniform(-2 * feed['u3'], 2 * feed['u3']),
        #                 'u2': np.random.uniform(0, 2 * feed['u2']),
        #
        #                }
        self.feed={}
        for k, v in feed.items():
            #if k in ['g1','g2','gL','tau_w']:
            if k in ['gL']:
            #if k not in ['I']:
                #self.feed[k]=torch.nn.Parameter(torch.tensor(init_values_rand[k],dtype=torch.float))
                self.feed[k] = torch.nn.Parameter(1.01 * torch.tensor(v, dtype=torch.float))
                self.register_parameter(k,self.feed[k])
            else:
                self.feed[k] = torch.tensor(v,dtype=torch.float)
        self.w0_learnable=torch.nn.Parameter(1.2*torch.tensor(0,dtype=torch.float))
        self.register_parameter('w0', self.w0_learnable)


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


#RCL penalize negative parameter values by augmenting the loss
def lossParameterOutofBounds(paramfeed):
    constraint_factor=1000 #1 seems better than 1000, at least for 10 pendulums
    loss = torch.tensor(0,dtype=torch.float)
    # g1 parameter bounds (see random initializations)
    loss += nn.functional.relu(-1.0 * paramfeed['g1'] + 3.6) + nn.functional.relu(paramfeed['g1'] - 4.4)
    # g2 parameter bounds (see random initializations)
    loss += nn.functional.relu(-1.0 * paramfeed['g2'] + 1) + nn.functional.relu(paramfeed['g2'] - 25)
    # gL parameter bounds (see random initializations)
    loss += nn.functional.relu(-1.0 * paramfeed['gL'] + 0.1) + nn.functional.relu(paramfeed['gL'] - 10)
    # tau_w parameter bounds (see random initializations)
    loss += nn.functional.relu(-1.0 * paramfeed['tau_w'] + 5) + nn.functional.relu(paramfeed['tau_w'] - 25)
    return constraint_factor*loss

def calcMAPE(paramfeed, truefeed):
    compare = []
    for name in ['g1','g2','gL','tau_w','u3','u4','u1','u2','V1','V2','VL']:
        diff = abs((float(paramfeed[name].data.numpy()) - truefeed[name]) / truefeed[name])
        compare.append(diff)
    return np.mean(compare)

def saveParameters(paramfeed):
    bestparams={}
    for name in ['g1','g2','gL','tau_w','u3','u4','u1','u2','V1','V2','VL']:
        bestparams[name] = paramfeed[name].item()
    return bestparams

#Simple spike counter (on membrane potential variable)
def spikeCounter(pred_y):
    spikecount=0
    for i in range(pred_y.shape[0]-1):
        if pred_y[i,0,0]<25.0 and pred_y[i+1,0,0]>25.0: #Upstroke
            spikecount+=1
    return spikecount

if __name__ == '__main__':

    ii = 0

    func = ODEFunc(args.randomseed_initialparams)
    #RCL Save initial parameter values
    init_params = saveParameters(func.feed)

    #RCL
    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    #optimizer = optim.Adam(func.parameters(), lr=1.5e-1)
    if args.lbfgs:
        optimizer = optim.LBFGS(func.parameters(), lr=2e-4)
    else:
        #optimizer = optim.Adam(func.parameters(), lr=1.5e-3)
        #optimizer = optim.SGD(func.parameters(), lr=5e-7)
        optimizer = optim.SGD(func.parameters(), lr=5e-9)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    #RCL Use MSE instead of abs
    lossfunc = torch.nn.MSELoss()

    #RCL
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=100,verbose=True)

    min_loss = 1e10
    best_params = None
    best_iter = None
    best_MAPE = None

    for itr in range(1, args.niters + 1):
        #with torch.no_grad():
            #pred_y_notbatch = odeint(func, true_y0, t, method=args.method)
            #batch_y0, batch_t, batch_y = get_batch()
            #w0 = w0_guess(true_y.squeeze().t()[0],
            #              func.feed['u1'].item(), func.feed['u2'].item(),
            #              func.feed['g1'].item(), func.feed['g2'].item(), func.feed['gL'].item(),
            #              func.feed['V1'].item(), func.feed['V2'].item(), func.feed['VL'], func.feed['I'].item(), dt)
            #w_g = w_guess(true_y.squeeze().t()[0], func.feed['tau_w'].item(), func.feed['u3'].item(), func.feed['u4'].item(), w0, dt)
            #batch_y0, batch_t, batch_y = get_batch_partialObs([0],pred_y_notbatch,w_g)
            #
            #batch_y0, batch_t, batch_y = get_batch_partialObs(w_g)

        optimizer.zero_grad()
        #pred_y = odeint(func, batch_y0.squeeze(), batch_t, method=args.method) #RCL had to call squeeze
        #print(pred_y, batch_y)
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #RCL Modified loss function, use MSE instead of abs
        #Use membrane potential only, slice and get 0-component of state

        #loss = lossfunc(pred_y[:,:,0],batch_y.squeeze()[:,:,0]) #RCL Had to call squeeze
        #loss = lossfunc(pred_y[:, :, 1], batch_y.squeeze()[:, :, 1])  # RCL Had to call squeeze

        #Add penalty for negative parameter values
        #loss += lossParameterOutofBounds(func.feed)
        #Add "regularization" for w(t)
        #loss += 10*torch.sum(pred_y[:,:,1]*pred_y[:,:,1]) #Not working as expected.

        #NO MINI-BATCHING!!! w_guess not needed???
        #true_y0[0,1]=func.w0_learnable
        init_state = torch.tensor([[0,0]], dtype=torch.float)
        init_state[0,0]=true_y0[0,0]
        init_state[0,1]=func.w0_learnable
        print('Initial state:',init_state)


        #pred_y = odeint(func, true_y0, t, method=args.method)
        pred_y = odeint(func, init_state, t, method=args.method)
        loss = lossfunc(pred_y[:, :, 0], true_y[:, :, 0]) + 10000*torch.nn.functional.relu(-func.w0_learnable)
        #loss = lossfunc(pred_y[:, :, 1], true_y[:, :, 1]) + 10000 * torch.nn.functional.relu(-func.w0_learnable)

        loss.backward()
        trainlossval = loss.item()

        #RCL Experiment with LBFGS
        #https: // pytorch.org / docs / stable / optim.html
        def closure():
            optimizer.zero_grad()
            #output = model(input)
            #pred_y_ = odeint(func, batch_y0.squeeze(), batch_t, method=args.method)
            #loss = loss_fn(output, target)
            #loss_ = lossfunc(pred_y_[:,:,0], batch_y.squeeze()[:,:,0])
            pred_y_ = odeint(func, init_state, t, method=args.method)
            loss_ = lossfunc(pred_y_[:, :, 0], true_y[:, :, 0]) + 10000 * torch.nn.functional.relu(-func.w0_learnable)
            loss_.backward()
            return loss_

        if args.lbfgs:
            optimizer.step(closure)  # Use for LBFGS
        else:
            optimizer.step() #Use for Adam

        #RCL
        #scheduler.step(loss)

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                #true_y0=true_y[0]
                #pred_y = odeint(func, true_y0, t, method=args.method)
                pred_y = odeint(func, init_state, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                #RCL Use MSE instead of abs
                #loss=lossfunc(pred_y[:,:,0],true_y[:,:,0])+lossParameterOutofBounds(func.feed)
                loss = lossfunc(pred_y[:, :, 0], true_y[:, :, 0])
                #loss = lossfunc(pred_y[:, :, 1], true_y[:, :, 1])
                totallossval=loss.item()
                avgm = calcMAPE(func.feed,feed)
                spikecount = spikeCounter(pred_y)
                print('Iter {:04d} | Total Loss {:.6f} | Train Loss Before Optimizer Update {:.6f} | MAPE {:.6f} | Spikes {:d}'.format(itr, totallossval, trainlossval, avgm, spikecount))
                #print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                #visualize(true_y, w_g, func, ii, plotV=False)
                #visualize(true_y, pred_y, func, ii, plotV=False)
                #RCL
                for param in func.parameters():
                    print(param.grad)
                #for param_group in optimizer.param_groups:
                #    print(param_group['lr'])
                ii += 1
                if totallossval<min_loss and spikecount==args.nspikes:
                    min_loss = totallossval
                    best_params = saveParameters(func.feed)
                    best_iter = itr
                    best_MAPE = avgm
                    torch.save(true_y, 'FG_ml_test3_rl_SimulateW_true_y_StableSGD.pt')
                    torch.save(pred_y, 'FG_ml_test3_rl_SimulateW_pred_y_StableSGD.pt')
                #print(func.feed)

        end = time.time()

    if best_params==None:
        print('No fit with {:d} spikes found'.format(args.nspikes))
    else:
        print('Seed:',args.randomseed_initialparams,'Loss:', min_loss, 'MAPE:', best_MAPE, 'Iter:', best_iter, 'Init:', init_params, 'Best:', best_params)
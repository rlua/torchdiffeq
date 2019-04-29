import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='ODE demo Pinsky-Rinzel\'s Reduced Traub model, 1 compartment')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4')
parser.add_argument('--data_size', type=int, default=2001)
parser.add_argument('--batch_time', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--lbfgs', action='store_true')


args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print('Running in adjoint mode (CAM)')
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

feed = {
    #maximal conductances in mS/cm^2
    'gL': 0.1, #leak
    'gNa': 30, #fast sodium
    'gKDR': 15, #delayed rectifier

    #reversal potentials (in mV)
    'VNa': 120,
    'VK': -15,
    'VL': 0,

    'Is': 0.25,

    #fraction of cable length assigned to soma (1-p for dendrite)
    'p': 0.5,

    #capacitance microF/cm^2
    'Cm': 3,
}

#init = [-4.6, 0.999, 0.001]
#true_y0=torch.tensor([init],dtype=torch.float)
#t = torch.linspace(0., 200., args.data_size)

def true_fromfile(filename):
    data=open(filename,'r').readlines()
    t=[]
    V=[]
    for line in data:
        cols=line.strip().split()
        if len(cols)<2:
            continue
        t.append(float(cols[0]))
        V.append([float(cols[1]),0,0])
    return len(t), torch.tensor(t), torch.tensor(V).unsqueeze(1)

#Load Fabrizio's time series for V(t)
args.data_size, t, true_y = true_fromfile('pr_test1_rl.txt')
print(args.data_size)
print(t)
print(true_y)

class Lambda(nn.Module):


    def forward(self, t, y):
        '''
        :param t:
        :param y:
        :return:

        Model from prsolve_t1_rk4.m
        %  Adapted from prsolve_rk4, 1 compartment model using the 4th order
        %  classic Runge-Kutta method
        %
        %  prsolve_t1_rk4.m
        %
        %  solve the pinsky-rinzel model truncated to 1 compartment
        %
        %  the state is y = [Vs h n]
        %  with:
        %
        %  Vs = soma potential
        %  h = fast sodium inactivation, n = delayed rectifier activation
        %
        %  usage   prsolve_t1_rk4(T,Is)     e.g.,  prsolve_t1_rk4(100,0)
        %
        %  where   T = duration of simulation (ms)
        %          Is = somatic current injection (microA/cm^2)

        %  figure produces Vs vs time
        %
        % Reference: [PR94] Pinsky PF, Rinzel J (1994) Intrinsic and Network
        % Rhythmogenesis in a Reduced Traub Model for CA3 Neurons. J Comp Neurosci
        % 1:39-60. Erratum in J Comp Neurosci 2:275, 1995.
        %

        dy = zeros(3,1);

        %somatic leak current
        Ils = gL*(y(1)-VL);

        %steady-state sodium activation (instantaneous)
        minf = am(y(1))/(am(y(1))+bm(y(1)));

        %sodium current (y(2) is h, inactivation of sodium current)
        INa = gNa*minf.^2.*y(2).*(y(1)-VNa);

        %delayed rectifier current (y(3) is n, activation of DR)
        IKDR = gKDR*y(3)*(y(1)-VK);

        %derivative update of somatic membrane potential, eq. 1 of [PR94]
        dy(1) = (-Ils - INa - IKDR + Is/p)/Cm;

        %derivative update of h, sodium inactivation
        dy(2) = ah(y(1))*(1-y(2)) - bh(y(1))*y(2);

        %derivative update of n, DR activation
        dy(3) = an(y(1))*(1-y(3)) - bn(y(1))*y(3);

        '''

        # For following rate constants, see eq. 6 of[PR94] and erratum
        # # forward rate constant for fast sodium
        # def am(v):
        #     return 0.32 * (13.1 - v) / (torch.exp((13.1 - v) / 4) - 1)
        #
        # # backward rate constant for fast sodium
        # def bm(v):
        #     return 0.28 * (v - 40.1) / (torch.exp((v - 40.1) / 5) - 1)
        #
        # # forward rate constant for DR activation
        # def an(v):
        #     return 0.016 * (35.1 - v) / (torch.exp((35.1 - v) / 5) - 1)
        #
        # # backward rate constant for DR activation
        # def bn(v):
        #     return 0.25 * torch.exp(0.5 - 0.025 * v)
        #
        # # forward rate constant for sodium inactivation
        # def ah(v):
        #     return 0.128 * torch.exp((17 - v) / 18)
        #
        # # backward rate constant for sodium inactivation
        # def bh(v):
        #     return 4. / (1 + torch.exp((40 - v) / 5))

        x=y.t()
        dx = torch.zeros_like(x)

        #The states are
        #Vs - x[0]
        #h - x[1]
        #n - x[2]

        #somatic leak current
        Ils = feed['gL']*(x[0]-feed['VL'])

        #steady - state sodium activation(instantaneous)
        am = 0.32 * (13.1 - x[0]) / (torch.exp((13.1 - x[0]) / 4) - 1)
        bm = 0.28 * (x[0] - 40.1) / (torch.exp((x[0] - 40.1) / 5) - 1)
        minf = am / (am + bm)

        #sodium current (x[1] is h, inactivation of sodium current)
        INa = feed['gNa']*minf*minf*x[1]*(x[0]-feed['VNa'])

        #delayed rectifier current (x[2] is n, activation of DR)
        IKDR = feed['gKDR']*x[2]*(x[0]-feed['VK'])

        #derivative update of somatic membrane potential, eq. 1 of [PR94]
        dx[0] = (-Ils - INa - IKDR + feed['Is']/feed['p'])/feed['Cm']

        #derivative update of h, sodium inactivation
        ah = 0.128 * torch.exp((17 - x[0]) / 18)
        bh = 4. / (1 + torch.exp((40 - x[0]) / 5))
        dx[1] = ah*(1-x[1]) - bh*x[1]

        #derivative update of n, DR activation
        an = 0.016 * (35.1 - x[0]) / (torch.exp((35.1 - x[0]) / 5) - 1)
        bn = 0.25 * torch.exp(0.5 - 0.025 * x[0])
        dx[2] = an*(1-x[2]) - bn*x[2]

        return dx.t()

# with torch.no_grad():
#     true_y = odeint(Lambda(), true_y0, t, method=args.method)
#
# #Add noise to observations
# noise_dist = torch.distributions.normal.Normal(0, 1.0)
# noise_shape = torch.Size([true_y.size()[0], true_y.size()[1], true_y.size()[2]])
# noise_samples = noise_dist.sample(sample_shape=noise_shape)
# true_y = true_y + noise_samples

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

#Simple spike counter (on membrane potential variable)
def spikeCounter(y):
    spikecount=0
    for i in range(y.shape[0]-1):
        if y[i,0,0]<50.0 and y[i+1,0,0]>50.0: #Upstroke
            spikecount+=1
    return spikecount

#Initially, I use this to get the index to the trace for the downstroke of the second spike and the
#upstroke of the third spike, assuming at least three spikes in the trace and sufficient
#resolution in the trace
def getInterval(y,threshold=80):
    upstrokecount=0
    downstrokecount=0
    i0=None
    i1=None
    i2=None
    i3=None
    for i in range(y.shape[0]-1):
        if y[i,0,0]<threshold and y[i+1,0,0]>threshold: #Upstroke
            upstrokecount+=1
            if upstrokecount == 1:
                i0 = i
            if upstrokecount == 2:
                i2 = i
        if y[i,0,0]>threshold and y[i+1,0,0]<threshold: #Downstroke
            downstrokecount+=1
            if downstrokecount == 1:
                i1 = i
            if downstrokecount == 2:
                i3 = i
    #Find the first spike peak between first upstroke and first downstroke
    imax1 = None
    Vmax = -1000
    for i in range(i0,i1):
        if y[i,0,0]> Vmax:
            Vmax=y[i,0,0]
            imax1=i
    #Find the second spike peak between 2nd upstroke and 2nd downstroke
    imax2 = None
    Vmax = -1000
    for i in range(i2,i3):
        if y[i,0,0]> Vmax:
            Vmax=y[i,0,0]
            imax2=i
    return imax1,imax2

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False) #111 one row, one column, first panel
    ax_phase = fig.add_subplot(122, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, t, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel(r'$t$')
        ax_traj.set_ylabel(r'$V_s$ (mV, somatic membrane potential)')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'r')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], 'r--')
        #ax_traj.plot(t[i1:i2].numpy(), true_y[i1:i2].numpy()[:, 0, 0], 'r')
        #ax_traj.plot(t[i1:i2].numpy(), pred_y[i1:i2].numpy()[:, 0, 0], 'r--')
        #ax_traj.plot(true_y.numpy()[:, 0, 0], 'r')
        #ax_traj.plot(pred_y.numpy()[:, 0, 0], 'r--')
        #ax_traj.set_xlim(t.min(), t.max())
        #ax_traj.set_xlim(t[i1:i2].min(), t[i1:i2].max())
        ax_traj.set_ylim(-20, 100)
        #ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel(r'$V_s$')
        ax_phase.set_ylabel(r'$h$')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-20, 100)
        ax_phase.set_ylim(0, 1)

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

def visualize_segments(true_y, pred_y_seg_list, iseg, t, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel(r'$t$')
        ax_traj.set_ylabel(r'$V_s$ (mV, somatic membrane potential)')
        ax_traj.plot(t.numpy()[iseg[0]:], true_y.numpy()[iseg[0]:, 0, 0], 'r')
        #ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], 'r--')
        for i in range(odefunc.ns):
            ax_traj.plot(t[iseg[i]:iseg[i+1]+1].numpy(), pred_y_seg_list[i][:, 0, 0], 'b--')
        #ax_traj.plot(t[i1:i2].numpy(), true_y[i1:i2].numpy()[:, 0, 0], 'r')
        #ax_traj.plot(t[i1:i2].numpy(), pred_y[i1:i2].numpy()[:, 0, 0], 'r--')
        #ax_traj.plot(true_y.numpy()[:, 0, 0], 'r')
        #ax_traj.plot(pred_y.numpy()[:, 0, 0], 'r--')
        #ax_traj.set_xlim(t.min(), t.max())
        #ax_traj.set_xlim(t[i1:i2].min(), t[i1:i2].max())
        ax_traj.set_ylim(-20, 100)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.01)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.feed={}
        for k, v in feed.items():
            if k in ['gL','gNa','gKDR']:
                self.feed[k]=torch.nn.Parameter(1.0*torch.tensor(v,dtype=torch.float))
                self.register_parameter(k,self.feed[k])
            else:
                self.feed[k] = torch.tensor(v,dtype=torch.float)

        #Number of segments to split the interval to fit
        self.ns=5
        h=[]
        n=[]
        # Make initial conditions of each segment optimizable (on unobservable h and n)
        #TODO use steady-state as initial values
        for i in range(self.ns):
            h.append(torch.nn.Parameter(1.0*torch.tensor(0,dtype=torch.float)))
            self.register_parameter('h%d' % i, h[i])
            n.append(torch.nn.Parameter(1.0*torch.tensor(0,dtype=torch.float)))
            self.register_parameter('n%d' % i, n[i])
        self.h=h
        self.n=n


    #Make sure to use self.feed in this forward method
    def forward(self, t, y):
        #y may contain multiple batches, take the transpose
        x=y.t()
        dx = torch.zeros_like(x)

        #somatic leak current
        Ils = self.feed['gL']*(x[0]-self.feed['VL'])

        #steady - state sodium activation(instantaneous)
        am = 0.32 * (13.1 - x[0]) / (torch.exp((13.1 - x[0]) / 4) - 1)
        bm = 0.28 * (x[0] - 40.1) / (torch.exp((x[0] - 40.1) / 5) - 1)
        minf =  am / (am + bm)

        #sodium current (x[1] is h, inactivation of sodium current)
        INa = self.feed['gNa']*minf*minf*x[1]*(x[0]-self.feed['VNa'])

        #delayed rectifier current (x[2] is n, activation of DR)
        IKDR = self.feed['gKDR']*x[2]*(x[0]-self.feed['VK'])

        #derivative update of somatic membrane potential, eq. 1 of [PR94]
        dx[0] = (-Ils - INa - IKDR + self.feed['Is']/self.feed['p'])/self.feed['Cm']

        #derivative update of h, sodium inactivation
        ah = 0.128 * torch.exp((17 - x[0]) / 18)
        bh = 4. / (1 + torch.exp((40 - x[0]) / 5))
        dx[1] = ah*(1-x[1]) - bh*x[1]

        #derivative update of n, DR activation
        an = 0.016 * (35.1 - x[0]) / (torch.exp((35.1 - x[0]) / 5) - 1)
        bn = 0.25 * torch.exp(0.5 - 0.025 * x[0])
        dx[2] = an*(1-x[2]) - bn*x[2]

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


def CustomWeightedLoss(pred,true):

    maxweight=true.shape[0]
    l=torch.tensor(0,dtype=torch.float)
    for i in range(true.shape[0]):
        weight=(maxweight - i)**2
        l += torch.mean((pred[i] - true[i]) * (pred[i] - true[i]) * weight)
    #return l/(true.shape[0])**3
    return l / (true.shape[0]) ** 2.5

if __name__ == '__main__':

    ii = 0

    func = ODEFunc()

    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    #optimizer = optim.Adam(func.parameters(), lr=1.5e-1)
    if args.lbfgs:
        optimizer = optim.LBFGS(func.parameters(), lr=2e-2)
    else:
        #optimizer = optim.Adam(func.parameters(), lr=1.5e-2)
        #optimizer = optim.SGD(func.parameters(), lr=1e-6)
        optimizer = optim.SGD(func.parameters(), lr=1e-5)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    lossfunc = torch.nn.MSELoss()

    i1,i2=getInterval(true_y) #From one peak to the next
    true_y_sub = true_y[i1:i2]
    t_sub = t[i1:i2]
    iseg=[]
    seglength=(i2-i1)//func.ns
    for i in range(func.ns):
        iseg.append(i1+seglength*i)
    iseg.append(i2)

    print('Fitting interval {} to {} split into {} segments'.format(i1,i2,func.ns))
    print('Segment starting indices at ',iseg)

    #RCL
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=100,verbose=True)

    for itr in range(1, args.niters + 1):
        #with torch.no_grad():
        #    pred_y_notbatch = odeint(func, true_y0, t, method=args.method)
        #    #batch_y0, batch_t, batch_y = get_batch()
        #    batch_y0, batch_t, batch_y = get_batch_partialObs([0],pred_y_notbatch)
        optimizer.zero_grad()
        #pred_y = odeint(func, batch_y0.squeeze(), batch_t, method=args.method) #RCL had to call squeeze
        #Use membrane potential only, slice and get 0-component of state
        #pred_y = odeint(func, true_y0, t, method=args.method)
        #loss = lossfunc(pred_y[:,:,0],batch_y.squeeze()[:,:,0]) #RCL Had to call squeeze
        #loss = lossfunc(pred_y[:, :, 0], true_y[:, :, 0])

        init_states=[]
        for i in range(func.ns):
            init_states.append(torch.tensor([[0,0,0]], dtype=torch.float))
            init_states[i][0,0] = true_y[iseg[i],0,0]
            init_states[i][0,1] = func.h[i]
            init_states[i][0,2] = func.n[i]
        print('Initial states:', init_states)

        pred_y_seg_numpylist=[]
        for i in range(func.ns):
            pred_y_seg = odeint(func, init_states[i], t[iseg[i]:iseg[i+1]+1], method=args.method)
            pred_y_seg_numpylist.append(pred_y_seg.detach().numpy())
            if i==0:
                loss = lossfunc(pred_y_seg[:, :, 0], true_y[iseg[i]:iseg[i+1]+1, :, 0])
            else:
                loss += lossfunc(pred_y_seg[:, :, 0], true_y[iseg[i]:iseg[i+1]+1, :, 0])

            #loss = CustomWeightedLoss(pred_y_sub[:, :, 0],true_y_sub[:, :, 0])

            #penalize deviations of h and n between end of one segment and beginning of another segment
            inextseg = (i+1)%func.ns
            loss += 150 * ((func.h[inextseg]-pred_y_seg[-1,0,1])**2 + (func.n[inextseg]-pred_y_seg[-1,0,2])**2)

            #penalize negative values of auxiliary variables
            loss += 10000*(torch.nn.functional.relu(-func.h[i])+torch.nn.functional.relu(-func.n[i]))

        #enforce periodicity in the auxiliary variables from one peak to the next
        #loss += 80 * ((func.h0-pred_y_sub[-1,0,1])**2 + (func.n0-pred_y_sub[-1,0,2])**2)

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
                #pred_y = odeint(func, true_y0, t, method=args.method)
                pred_y = odeint(func, init_states[0], t[i1:], method=args.method)
                #loss=lossfunc(pred_y,true_y[i1:])
                #pred_y_sub = odeint(func, init_state, t_sub, method=args.method)
                loss = lossfunc(pred_y[:, :, 0], true_y[i1:, :, 0])
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                visualize(true_y[i1:], pred_y, t[i1:], func, ii)
                #visualize(true_y_sub, pred_y_sub, t[i1:], func, ii)
                #visualize_segments(true_y, pred_y_seg_numpylist, iseg, t, func, ii)

                #for param in func.parameters():
                #    print(param.grad)
                #for param_group in optimizer.param_groups:
                #    print(param_group['lr'])
                ii += 1

        end = time.time()

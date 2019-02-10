import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='ODE demo Coupled Pendulums with partial observations, random parameter initialization, noise, penalties on negative parameters, MAPE calculation, find best parameter estimates using total loss.')
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'adams'], default='rk4') #RCL modified default
parser.add_argument('--data_size', type=int, default=201) #RCL modified default
parser.add_argument('--batch_time', type=int, default=20) #RCL modified default
parser.add_argument('--batch_size', type=int, default=10) #RCL modified default
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=1) #RCL modified default
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--npendulums', type=int, default=4) #RCL added new option
parser.add_argument('--nstatesobserved', type=int, default=8) #RCL added new option
parser.add_argument('--randomseed_initialparams', type=int, default=0) #RCL added new option
parser.add_argument('--position_noise_stddev', type=float, default=0) #RCL added new option
parser.add_argument('--velocity_noise_stddev', type=float, default=0) #RCL added new option
args = parser.parse_args()

#RCL
assert args.npendulums <= 1000, 'npendulums in the model must not be greater 1000'
N_=args.npendulums
N2_=2*N_
print('Number of pendulums and state variables:', N_, N2_)
print('Number of observable states during training:', args.nstatesobserved)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print('Running in adjoint mode (CAM)')
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
     9.28963387,   9.04412782,   8.11297509,  10.57329466,
   # Generate another 900 for a total of 1000 k's, with x=scipy.stats.truncnorm.rvs(-3,3,size=900) and x+(2*3.14159*0.5)**2
   11.21978935, 9.944444, 10.26826134, 10.32178418, 8.88788075,
   10.39337106, 9.4609917, 7.95759544, 12.02439173, 8.81780138,
   10.51421799, 8.93842172, 10.75618284, 10.03792469, 10.74323534,
   10.88525288, 10.35791491, 10.53802695, 9.74206714, 11.36689012,
   11.67427287, 9.34978366, 9.74506626, 10.00769291, 8.1518902,
   10.64063616, 9.00840935, 10.93408021, 9.91396042, 8.40062681,
   9.42616739, 10.3088485, 12.64725444, 11.26912372, 10.36521387,
   10.81062121, 11.79604989, 9.86659922, 9.2848802, 10.46472384,
   9.56520284, 8.23660203, 10.12613568, 8.79578855, 10.82584808,
   10.03527012, 9.49166719, 9.21972768, 11.5363978, 10.16827745,
   9.92278418, 10.20337829, 9.40726184, 9.64524946, 9.65084392,
   8.80497612, 12.28722582, 8.53876998, 7.8362678, 11.36834893,
   11.16991559, 10.04294516, 8.38377732, 9.39039069, 9.11959451,
   11.12467837, 7.72205236, 8.14370212, 10.28517594, 9.49283302,
   10.15696934, 8.03414443, 10.21555939, 10.83526388, 11.16956729,
   10.77301344, 11.37661089, 9.17368903, 8.16472567, 11.3885983,
   9.11808581, 8.92658806, 10.20336429, 9.57778288, 9.16487285,
   10.90372375, 8.44249457, 9.40888403, 10.67298215, 9.17268191,
   11.36005118, 8.83224579, 8.9966006, 8.52550978, 11.20227607,
   8.52437323, 9.93045147, 8.03863105, 8.11372746, 11.38876494,
   10.56869854, 11.06826732, 11.09554358, 9.59122597, 10.06684786,
   9.29182535, 9.23353378, 9.82029918, 9.53758056, 10.2787273,
   9.08155829, 9.20087045, 10.21255219, 9.38880508, 8.93627773,
   10.22452719, 10.22166596, 9.88156174, 9.09825525, 8.84361232,
   9.2133729, 9.20028982, 10.8740542, 9.18594162, 9.78876997,
   10.43553449, 9.64884917, 10.15887014, 8.56977044, 11.36045373,
   9.17016994, 9.99638906, 9.14158081, 11.87128758, 9.90918302,
   10.69670224, 10.51103437, 9.83111661, 8.22024243, 8.09000772,
   11.00904789, 9.01615037, 9.4501872, 10.37203603, 10.73605479,
   9.47776522, 9.38712222, 9.03537363, 10.14345839, 9.81404918,
   10.51201643, 7.66025911, 10.96911762, 9.82689456, 9.79950605,
   9.46764531, 8.89367225, 9.17667969, 10.28676101, 10.22146124,
   9.02584242, 8.56443183, 10.93760474, 9.48843144, 8.81062641,
   10.43750798, 9.04325349, 8.4670543, 10.62007366, 10.28972668,
   9.25221965, 9.70985357, 10.35167502, 11.66331827, 10.33727523,
   11.2301012, 10.30715493, 9.06922232, 9.57891052, 11.45746086,
   9.27300712, 9.31180642, 10.44960765, 10.86652709, 9.91367562,
   9.8670779, 9.85954726, 10.6029772, 9.76823808, 9.95089677,
   10.31903144, 10.56699008, 10.96426489, 10.29432619, 9.42496993,
   9.44107973, 10.14235461, 10.08027592, 11.28355843, 7.64642568,
   9.70712161, 9.35541101, 9.01006936, 8.52873637, 10.21235247,
   10.59193294, 8.6648071, 10.86393348, 10.02735545, 10.54529515,
   9.94913741, 7.76660953, 10.39697976, 11.97080282, 9.39290456,
   10.57341322, 8.72415178, 9.82526824, 7.34697778, 11.65894676,
   10.90145866, 12.04399715, 10.63344032, 7.7548648, 9.68066411,
   10.45334559, 10.1261416, 9.61106683, 8.59278577, 11.04577494,
   8.95614076, 8.06919558, 9.2637512, 8.50551953, 10.10644171,
   10.55023783, 9.64227002, 9.72194075, 8.22533961, 8.16514186,
   10.06879009, 9.90835694, 9.89044464, 8.53881827, 10.132576,
   12.14780414, 8.58421276, 11.192444, 8.41129639, 9.0269955,
   9.04520818, 10.60373472, 11.2465724, 11.24690529, 9.97368391,
   9.6234591, 10.64507241, 8.82699919, 11.7456576, 9.09361579,
   10.6712381, 10.33887423, 10.13194582, 10.98576224, 10.23004037,
   9.84380922, 10.4728452, 8.83025071, 10.14033696, 11.76687679,
   11.04294289, 8.81516834, 10.90797756, 8.81424524, 10.57862566,
   10.36376741, 9.46912813, 11.14899836, 9.49154885, 10.00143834,
   8.84619027, 10.50451005, 10.34336426, 10.41038654, 10.91807949,
   9.87938217, 9.31863212, 11.50695649, 9.56919773, 10.60236862,
   8.87833241, 10.31450413, 9.94356406, 11.29911383, 8.91896124,
   10.94954069, 10.83987133, 10.497164, 11.61435798, 10.32077088,
   9.44316782, 9.6496446, 10.68492109, 10.1179429, 10.03440074,
   7.79990125, 10.06976192, 9.93672717, 10.06310987, 10.81936581,
   11.29992494, 12.30197785, 9.36953952, 10.45652539, 11.09841053,
   11.51303969, 9.46229851, 8.20141396, 11.66801922, 10.38038817,
   9.303474, 8.87047267, 11.97184184, 12.68200543, 9.56001539,
   8.62627901, 9.50193582, 9.74813982, 9.54115669, 10.95853241,
   9.64410675, 10.53234195, 10.24799028, 9.48571302, 9.19084375,
   10.2955305, 9.45441886, 9.72829976, 8.62813811, 11.85997822,
   10.32044806, 12.43541786, 9.72608927, 8.49696781, 8.76119818,
   9.60810456, 7.94966328, 8.67896061, 9.73193919, 10.89780676,
   9.81282473, 9.32673397, 9.92504801, 10.67909471, 9.31946024,
   10.65041989, 8.83150411, 9.32778997, 11.04457108, 10.75437151,
   10.86967667, 10.17634174, 8.91495948, 9.48427114, 10.78791438,
   9.0822349, 11.06496821, 10.10715191, 9.22906202, 10.5430121,
   10.06018154, 10.74173072, 10.89941318, 8.44363985, 10.53768514,
   8.73920473, 9.41828157, 10.40358334, 9.93500794, 10.75057379,
   9.21370598, 9.87801537, 8.52039315, 9.2365083, 10.06985125,
   7.76011669, 9.74400045, 9.61472871, 10.63784183, 9.66003535,
   10.67559083, 10.19628464, 9.5465563, 12.11528444, 9.42870902,
   9.37196313, 10.3169786, 9.98388433, 10.11851802, 10.35743918,
   10.99419073, 9.7974141, 10.67714546, 11.64397714, 9.31539497,
   9.81292472, 8.65913427, 10.36008744, 9.43328155, 9.15225933,
   11.1538854, 9.46298513, 8.40901034, 10.571164, 9.54691923,
   10.57310783, 8.36773492, 11.21779953, 10.14885844, 10.95432212,
   9.85333541, 9.00967162, 7.90704959, 9.45228379, 9.25830498,
   11.14277417, 8.96922092, 9.76135877, 7.71589597, 9.60327066,
   10.96114914, 9.92298622, 9.01120526, 9.43113, 10.05997623,
   9.68580846, 9.47722336, 9.85469657, 9.90399174, 8.77914595,
   9.3321623, 8.43492063, 9.57947178, 10.6878517, 12.6571476,
   8.54997614, 10.98489089, 9.36358335, 8.9008215, 11.5242973,
   10.64134361, 11.15792047, 9.64430257, 10.24448992, 9.68909944,
   10.46975494, 8.8443774, 10.39857989, 8.55975151, 9.88940929,
   8.5309243, 9.48041776, 8.80090199, 9.48814001, 11.26395456,
   9.52533575, 8.91766285, 8.11368622, 12.57501737, 10.31164857,
   10.70139791, 10.37856346, 9.89086848, 9.88895254, 10.21433114,
   10.97071046, 9.49362845, 8.98363614, 8.10293007, 10.42200032,
   10.20886465, 9.34350766, 10.11093804, 10.70417311, 11.9765642,
   8.58330721, 9.99680675, 9.80912971, 8.94678202, 10.37734039,
   8.75039506, 10.18573794, 8.93006597, 9.57558566, 10.96316471,
   9.35253803, 9.98638235, 9.52867531, 10.88942635, 9.46525589,
   9.37762719, 10.4864597, 11.95507795, 10.32070715, 10.83206548,
   9.40280966, 9.60042819, 9.12521946, 9.87869342, 11.05154244,
   10.03781901, 10.07926871, 9.60661407, 10.09294672, 10.7075981,
   9.38226957, 8.78447219, 11.25194629, 10.73463298, 8.48054286,
   9.09421068, 10.25815823, 8.46709965, 10.4442301, 9.3259769,
   8.7424962, 9.33855193, 10.33433755, 9.42832968, 12.13880542,
   9.69716261, 10.23867805, 9.00409608, 10.07228003, 9.82551212,
   9.35761089, 8.6919673, 11.54084996, 10.42125875, 11.14235911,
   10.29933862, 9.82553056, 10.24240037, 8.77191838, 9.64989178,
   11.73499329, 9.01865897, 9.75167975, 9.14341675, 9.32124662,
   9.0284932, 9.13168481, 10.67167671, 10.49360047, 10.04923819,
   9.63551495, 11.21949822, 9.45149076, 10.41514279, 10.24678875,
   8.24955176, 11.44331788, 10.09499287, 9.68730034, 9.19343085,
   10.65257767, 10.80986463, 10.5221653, 8.33978842, 8.74477537,
   8.2608277, 9.54193734, 9.64984904, 8.65376148, 10.69618296,
   10.75269429, 9.64673076, 7.3932546, 9.44009114, 10.37128438,
   9.79260349, 10.07405127, 9.63647745, 10.69625044, 12.31205639,
   9.11552381, 11.05842707, 9.63686403, 8.55591111, 10.31460674,
   10.78049455, 10.97156863, 10.66899253, 9.95752047, 10.86729577,
   10.26074643, 11.78261378, 8.94406819, 8.2655695, 10.16992595,
   9.10211078, 9.30392658, 9.31328891, 10.25100107, 8.7527048,
   9.49105748, 7.69550048, 10.39467262, 9.86145745, 11.04134083,
   9.64156551, 10.28904423, 11.8240103, 9.50096587, 9.34393032,
   10.71427001, 10.3367092, 11.62771416, 10.62442818, 10.06472056,
   10.79112555, 10.6242089, 9.32499962, 10.26779113, 10.21033681,
   10.49110201, 10.31501447, 9.61343859, 9.50929873, 9.00445887,
   10.85887457, 8.94987169, 9.78471434, 10.98015545, 7.44238256,
   8.09611109, 10.54444903, 11.46003003, 9.17306318, 9.44948731,
   9.82389716, 9.86194934, 10.79391177, 11.41524065, 9.13838369,
   9.94598638, 10.11551447, 11.07681109, 9.40423005, 10.27751764,
   9.52123014, 10.57709868, 11.03521007, 8.29805484, 10.45219402,
   10.59775893, 10.99933939, 10.82134691, 10.17111418, 9.91450973,
   9.50990971, 10.82108903, 10.52602253, 10.01294257, 9.95877756,
   8.49949205, 10.69067313, 10.61371919, 9.38012946, 10.63331853,
   10.17688591, 9.6055093, 11.77582242, 10.96402252, 9.95210755,
   9.22481828, 12.17427973, 11.03275241, 9.88214128, 11.53209479,
   7.73840973, 9.78831142, 10.595896, 7.3503907, 9.34761532,
   9.81696178, 9.19339613, 9.64878495, 10.83454697, 11.56416514,
   11.22730333, 10.78928475, 9.21495155, 8.53465941, 8.82358927,
   10.95593961, 9.7632807, 9.37215355, 9.82930452, 9.71624547,
   9.47185658, 10.36049064, 11.33814925, 10.14219435, 10.33571833,
   11.39032867, 6.89698682, 9.65873235, 8.2755103, 9.71921893,
   9.77776202, 9.70858703, 9.05821421, 9.16816756, 8.47565619,
   12.28130281, 9.30294761, 10.31229748, 9.67000576, 11.09173955,
   7.95213722, 8.96228858, 9.98977366, 9.2046409, 10.04051058,
   9.64216228, 9.76849576, 10.97303239, 9.86038346, 11.32666326,
   9.28557905, 9.53258791, 9.74956296, 9.24627464, 10.2141172,
   12.01515868, 8.90106568, 11.81825306, 9.68496282, 10.56164807,
   8.55608435, 9.53157947, 9.22685791, 9.58230897, 10.4722998,
   10.84339707, 11.30711651, 10.46506239, 10.05272239, 10.11369186,
   9.81663009, 11.28262861, 10.26306327, 10.10320744, 9.09924086,
   9.40750975, 10.2480309, 9.80370068, 8.03400819, 9.28629504,
   9.06243448, 10.80544923, 8.94568824, 8.66803073, 11.28605903,
   11.44341166, 10.47771079, 9.86324127, 8.96850112, 8.74814984,
   10.54199334, 8.84648525, 11.66141197, 10.38911691, 10.35829693,
   10.69949087, 10.78295, 11.31839091, 10.22074682, 9.18699715,
   10.01312285, 11.5268666, 12.0310883, 10.08808712, 9.07267667,
   10.07182804, 11.11708944, 9.28067419, 9.9566318, 8.58969145,
   9.83347852, 8.92169354, 10.7198553, 9.7462605, 11.30735271,
   9.15561129, 11.35885819, 9.17130121, 11.54796168, 9.93238969,
   11.48264213, 10.28373691, 8.94434328, 10.11189769, 9.26265783,
   10.52199472, 9.73752908, 11.29172837, 10.2178304, 10.10322344,
   9.19298433, 9.13335738, 9.26432868, 10.22541, 9.49097643,
   9.70469644, 10.85737274, 12.10073598, 9.0101263, 10.08467167,
   8.76169495, 8.63717064, 9.23658062, 9.11423151, 9.91325722,
   8.1877674, 11.3356499, 10.58043551, 9.371881, 9.90778234,
   10.66956271, 9.40418226, 12.00896467, 9.49398012, 10.47153522,
   10.36844065, 8.49062903, 9.35690983, 11.0997447, 10.33252682,
   11.26677252, 11.13511469, 11.78974788, 8.7743021, 9.6160213,
   10.03573477, 9.38468479, 8.65612262, 10.43789178, 9.65803863,
   11.16758662, 12.11595283, 10.25544377, 7.45072426, 10.6813838,
   8.74249448, 9.62148808, 11.1215495, 8.8306008, 10.73683587,
   10.04109909, 9.8799933, 10.150334, 9.84540155, 11.37778919,
   9.85765243, 9.14322838, 9.88456159, 9.97253098, 11.98321155,
   9.46146745, 8.84862252, 9.687911, 8.99708494, 9.31624027,
   8.22113171, 9.50341823, 8.18138328, 10.15876417, 10.6422669,
   7.659341, 10.16363942, 10.120822, 8.18359876, 11.09086425,
   8.77394124, 8.50409776, 9.81379528, 8.59606607, 10.33109229,
   12.03504499, 11.53486159, 8.39741477, 8.60731311, 9.54110448,
   9.75653925, 10.7141879, 8.77554357, 10.20227057, 10.5160801]

#feed={'rho':4.0}
feed={}
for i in range(N_):
    var_name='k%d' % (i+1)
    feed[var_name]=k_groundtruth[i]
    rho_name = 'r%d' % (i + 1)
    feed[rho_name]=4.0
print('Number of parameters:',len(feed))

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
            #dx[ivel]=-feed['k%d'%(i+1)]*torch.sin(x[ipos])-feed['rho']*(2*x[ipos]-x[iposnext]-x[iposprev])
            #Make r (rhos) different
            iprev=i
            if i==0:
                iprev=N_
            dx[ivel] = -feed['k%d' % (i + 1)] * torch.sin(x[ipos]) \
                       -feed['r%d'%(i+1)] * (x[ipos] - x[iposnext]) \
                       -feed['r%d'%iprev] * (x[ipos] - x[iposprev])

        return dx.t()

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method=args.method)

#RCL Set an explicit seed for noise for reproducibility
torch.manual_seed(0)
m = torch.distributions.normal.Normal(0, args.position_noise_stddev)
m1 = torch.distributions.normal.Normal(0, args.velocity_noise_stddev)
#Add noise to observations
#print(true_y.size()[0], type(true_y.size()[0]))
h = torch.Size([true_y.size()[0], true_y.size()[1], int(true_y.size()[2]/2)])
h1 = torch.Size([true_y.size()[0], true_y.size()[1], true_y.size()[2]])
samples_1 = m.sample(sample_shape=h)
samples_2 = m1.sample(sample_shape=h1)
samples_2[:,:,:-1:2] = samples_1
true_y = true_y + samples_2


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
    batch_y0[:,:,idx] = true_y[s][:,:,idx] #Uncomment this to initialize batches using combination of states from simulated and ground truths
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
        ax_traj.set_ylabel(r'$\phi$,$v$')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        #RCL Modified limits for CP
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend([r'$\phi(t)$ ground truth',r'$v(t)$ ground truth',r'$\phi(t)$ fit',r'$v(t)$ fit'])

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel(r'$\phi(t)$')
        ax_phase.set_ylabel(r'$v(t)$')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        #RCL Modified limits for CP
        ax_phase.set_xlim(-0.5, 0.5)
        ax_phase.set_ylim(-2, 2)
        ax_phase.legend(['ground truth','fit'])

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

    def __init__(self, randomseed_initialparams):
        super(ODEFunc, self).__init__()

        #RCL
        np.random.seed(randomseed_initialparams)
        self.feed={}
        for k, v in feed.items():
            m = False
            while m == False:
                multiplier = float(np.random.lognormal(0, 0.5, 1))
                if 1.5 <= multiplier <=2.5:
                    m = True
                elif 0 < multiplier <= 0.6:
                    m = True
            #multiplier=1.1 #Testing
            self.feed[k]=nn.Parameter(multiplier*torch.tensor(v))
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
            #dx[ivel]=-self.feed['k%d'%(i+1)]*torch.sin(x[ipos])-self.feed['rho']*(2*x[ipos]-x[iposnext]-x[iposprev])
            iprev=i
            if i==0:
                iprev=N_
            dx[ivel] = -self.feed['k%d' % (i + 1)] * torch.sin(x[ipos])  \
                       -self.feed['r%d'%(i+1)] * (x[ipos] - x[iposnext]) \
                       -self.feed['r%d'%iprev] * (x[ipos] - x[iposprev])
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
    loss=torch.tensor(0,dtype=torch.float)
    for k,v in paramfeed.items():
        loss+=nn.functional.relu(-v)
    return constraint_factor*loss

def calcMAPE(paramfeed, truefeed):
    compare = []
    #diff = abs((float(paramfeed['rho'].data.numpy()) - truefeed['rho']) / truefeed['rho'])
    #compare.append(diff)
    for mm in range(N_):
        diff = abs((float(paramfeed['k%d' % (mm + 1)].data.numpy()) - truefeed['k%d' % (mm + 1)]) / truefeed['k%d' % (mm + 1)])
        compare.append(diff)
        diff = abs((float(paramfeed['r%d' % (mm + 1)].data.numpy()) - truefeed['r%d' % (mm + 1)]) / truefeed['r%d' % (mm + 1)])
        compare.append(diff)
    return np.mean(compare)

def saveParameters(paramfeed):
    bestparams={}
    for k,v in paramfeed.items():
        bestparams[k]=v.item()
    return bestparams


if __name__ == '__main__':

    ii = 0
    end1 = time.time()
    func = ODEFunc(args.randomseed_initialparams)
    #RCL Save initial parameter values
    init_params = saveParameters(func.feed)

    # RCL
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    #optimizer = optim.RMSprop(func.feed.values(), lr=1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1.0e-1)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    #Index for sampling partial observability
    #RCL Seed this and make the fraction of observed a parameter
    # and select randomly without replacement
    #idx = np.random.randint(2*args.npendulums, size=args.npendulums) #randint is with replacement
    np.random.seed(0)
    idx = np.random.choice(N2_, size=args.nstatesobserved, replace=False)  #This is without replacement
    print('Random states selected as observed', idx)
    #RCL Use MSE instead of abs
    lossfunc = nn.MSELoss()

    min_loss = 1e10
    best_params = None
    best_iter = None
    best_MAPE = None

    for itr in range(1, args.niters + 1):
        end = time.time()
        #RCL
        with torch.no_grad():
            pred_y_notbatch = odeint(func, true_y0, t, method=args.method)
            #batch_y0, batch_t, batch_y = get_batch() #This one uses the ground truth of all states for initial values, so not appropriate when observations are partial
            batch_y0, batch_t, batch_y = get_batch_partialObs(idx,pred_y_notbatch)
        optimizer.zero_grad()
        #batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0.squeeze(), batch_t, method=args.method) #RCL had to call squeeze
        #loss = torch.mean(torch.abs(pred_y - batch_y))
        #RCL Modified loss function, use MSE instead of abs
        loss = lossfunc(pred_y[:,:,idx],batch_y.squeeze()[:,:,idx]) #RCL Had to call squeeze
        #Add penalty for negative parameter values
        loss += lossParameterOutofBounds(func.feed)
        loss.backward()
        #RCL
        trainlossval = loss.item()
        #if trainlossval<min_loss:
        #    #Use the training batch loss instead of Total Loss ("lossfunc(pred_y,true_y)") because this was used in the paper and maintain continuity with previous work
        #    #Using Total Loss probably better
        #    min_loss=trainlossval
        #    best_params = saveParameters(func.feed)
        #    best_iter = itr - 1
        #    best_MAPE = calcMAPE(func.feed,feed)

        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                #RCL Use MSE instead of abs
                loss=lossfunc(pred_y,true_y)+lossParameterOutofBounds(func.feed)
                totallossval=loss.item()
                avgm = calcMAPE(func.feed,feed)
                print('Iter {:04d} | Total Loss {:.6f} | Train Loss Before Optimizer Update {:.6f} | MAPE {:.6f}'.format(itr, totallossval, trainlossval, avgm))
                visualize(true_y, pred_y, func, ii)
                ii += 1
                if totallossval<min_loss:
                    min_loss=totallossval
                    best_params = saveParameters(func.feed)
                    best_iter = itr
                    best_MAPE = avgm
                print(func.feed)

        ##end = time.time()
        #print("Time Per Iteration: ", time.time() - end)


    print("TOTAL TIME: ", time.time() - end1)

    #Print Seed: 100, Loss: 0.002408005530014634, MAPE: XX, Iter: 99, Init: {'k1': 3.7224637319738574, ... }. Best: {...},
    print('Seed:',args.randomseed_initialparams,'Loss:', min_loss, 'MAPE:', best_MAPE, 'Iter:', best_iter, 'Init:', init_params, 'Best:', best_params)
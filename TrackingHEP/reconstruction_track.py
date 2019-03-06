
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import errno
import argparse
from collections import Counter
from tqdm import tqdm, tqdm_notebook
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as utils_data
import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description='Runs NN model')
parser.add_argument('--N', type=int, default=10, required=False)
parser.add_argument('--M',type=int, default=4,required=False)

# N : vertices
# M: maxtracks
# E: epochs
#B: BATCH_SIZE
EVENTS = 100
args = parser.parse_args()
M = args.M
N = args.N

mt0 = M
mt1 = M
mt2 = M
LPRED = 1 

MAX_TRACKS = M

z2polar = lambda z: [ abs(z[0] + 1j * z[1]), np.angle(z[0] + 1j * z[1]) ]

def distphi(x, y):
    dimensions = 2 * np.pi
    delta = y - x
    delta = np.where(np.abs(delta) > 0.5 * dimensions, delta - np.sign(delta) * dimensions, delta)
    return delta

def distphi2(x, y):
    dimensions = 2 * np.pi
    delta = y - x
    delta = np.where(np.abs(delta) > 0.5 * dimensions, delta - np.sign(delta) * dimensions, delta)
    if delta < 1:
        return delta**2
    else:
        return 999


def find_n_closer(values, n):
    idx = np.argpartition(np.abs(values), n)
    if len(values.shape) == 2:
        n_close = values[np.arange(values.shape[0])[:, None], idx[:,:n]]
    elif len(values.shape) == 1:
        n_close = values[idx[:n]]
    else:
        return('max two d array')
    n_close.sort()
    return n_close

def select_els(row):
    pos = row[row >= 0]
    neg = row[row < 0]
    pos = find_n_closer(pos, MAX_TRACKS // 2 + 1)
    neg = find_n_closer(neg, MAX_TRACKS // 2)
    return np.concatenate([neg, pos])

    #filename_train ="hits_" + "v"+ str(N) + "_e" + str(EVENTS) + "_" + str(1) + ".csv"
filename_test  ="hits_" + "v"+ str(N) + "_e" + str(EVENTS) + "_" + str(2) + "_new.csv"
print(filename_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

sim_data_test = pd.read_csv(filename_test, header=None)
print(sim_data_test.shape[1])
sim_data_test.columns  = ['event', 'vertex', 'particle', 'layer', 'iphi', 'x', 'y', 'cluster_id','hit_id','phi']
#sim_data_test

class Track_Linear1(nn.Module):
    #Inside the init function, we initialize any layers
    def __init__(self, mt0,mt1,mt2, lin_h_o):
        super(Track_Linear1, self).__init__()
        
        #mt0 = 16
        self.mt0 = mt0
        #mt1 = 16
        self.mt1 = mt1
        #mt2 = 16
        self.mt2 = mt2
        self.lin_h_o = lin_h_o
        self.lin_h = self.lin_h_o * 2
        self.lin2_h = self.lin_h_o * 3 * 2


        #self.mt0 = mt0 #...
        
        self.linear1a0 = nn.Linear(self.mt0, self.lin_h)
        self.linear2a0 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a0 = nn.Linear(self.lin_h + 1, self.lin_h_o)
        
        self.linear1a1 = nn.Linear(self.mt1, self.lin_h)
        self.linear2a1 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a1 = nn.Linear(self.lin_h + 1, self.lin_h_o)
        
        self.linear1a2 = nn.Linear(self.mt2, self.lin_h)
        self.linear2a2 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a2 = nn.Linear(self.lin_h + 1, self.lin_h_o)
        
        self.linear = nn.Linear(self.lin_h_o * 3, self.lin2_h)        
        self.linear2 = nn.Linear(self.lin2_h, self.lin2_h // 2 + 1)
        self.linear3 = nn.Linear(self.lin2_h // 2 + 1, self.lin_h_o)
        self.linear4 = nn.Linear(self.lin_h_o, 1)

    def forward(self, x0, x1, x2):
        
    #if self.mt0>0:
        h0 = self.linear1a0(x0)
    #   h0 = F.relu(h0)        
        h0 = self.linear2a0(h0)
    #   h0 = F.relu(h0)
        h0 = self.linear3a0(h0)
    #   h0 = F.relu(h0)
    #    else:
    #        h0 = x0

        h1 = self.linear1a1(x1)
#        h1 = F.relu(h1)
        h1 = self.linear2a1(h1)
#        h1 = F.relu(h1)
        h1 = self.linear3a1(h1)
#        h1 = F.relu(h1)

        h2 = self.linear1a2(x2)
#        h2 = F.relu(h2)
        h2 = self.linear2a2(h2)
#        h2 = F.relu(h2)
        h2 = self.linear3a2(h2)
#        h2 = F.relu(h2)
        
        h = torch.cat([h0, h1, h2], dim=1)
        x_out = self.linear(h)
        x_out = F.relu(x_out)
#        x_out = F.dropout(x_out, 0.2, training=self.training)
        x_out = self.linear2(x_out)
        x_out = F.relu(x_out)
#        x_out = F.dropout(x_out, 0.2, training=self.training)
        x_out = self.linear3(x_out)
        x_out = F.relu(x_out)
#        x_out = F.dropout(x_out, 0.2, training=self.training)
        x_out = self.linear4(x_out)
        return x_out

class Track_LinearNext(nn.Module):
    def __init__(self, mt1, mt2, lin_h_o, len_yd):
        super(Track_LinearNext, self).__init__()

        self.mt1 = mt1
        self.mt2 = mt2
        self.lin_h_o = lin_h_o
        self.lin_h = self.lin_h_o * 2
        self.lin2_h = self.lin_h_o * 2 * 2
        
        self.len_yd = len_yd
        
        self.linear1a1 = nn.Linear(self.mt1, self.lin_h)
        self.linear2a1 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a1 = nn.Linear(self.lin_h + 1, self.lin_h_o)

        self.linear1a2 = nn.Linear(self.mt2, self.lin_h)
        self.linear2a2 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a2 = nn.Linear(self.lin_h + 1, self.lin_h_o)
        
        self.linear = nn.Linear(self.lin_h_o * 2, self.lin2_h)        
        self.linear2 = nn.Linear(self.lin2_h, self.lin2_h // 2 + 1)
        self.linear3 = nn.Linear(self.lin2_h // 2 + 1, self.lin_h_o)

        self.lin_last = self.lin_h_o + self.len_yd
        self.linear4 = nn.Linear(self.lin_last, self.lin_last * 2 + 1)
        self.linear5 = nn.Linear(self.lin_last * 2 + 1, self.lin_last // 2  + 1)
        self.linear6 = nn.Linear(self.lin_last // 2 + 1, 1)
            
    def forward(self, x1, x2, yd):  
        
        h1 = self.linear1a1(x1)
        h1 = self.linear2a1(h1)
        h1 = self.linear3a1(h1)

        h2 = self.linear1a2(x2)
        h2 = self.linear2a2(h2)
        h2 = self.linear3a2(h2)
         
        h = torch.cat([h1, h2], dim=1)
        x_out = self.linear(h)
        x_out = F.relu(x_out)
        x_out = self.linear2(x_out)
        x_out = F.relu(x_out)
        x_out = self.linear3(x_out)
    
        x_last = torch.cat([x_out, yd], dim=1)
        x_last = self.linear4(x_last)
        x_last = F.relu(x_last)
        x_last = self.linear5(x_last)
        x_last = F.relu(x_last)
        x_last = self.linear6(x_last)
        
        return x_last


filename = 'MODEL_v50_e100_mt_'+str(M)+'_'+str(M)+'_B50_E2000/MODEL_v50_e50_mt'+str(M)+'_'+str(M)+'_'+str(M)+'_B50_E2000_'
print(filename)


track_linear1 = Track_Linear1(mt0+1, mt1+1, mt2 +1,8 ).double().to(device)
track_linear1.load_state_dict(torch.load(filename+'L1.h5',map_location='cpu'))

track_linear2 = Track_LinearNext(mt1+1,mt2+1, 8, 1).double().to(device)
track_linear2.load_state_dict(torch.load(filename+'L2.h5',map_location='cpu'))

track_linear3 = Track_LinearNext(mt1+1,mt2+1, 8, 2).double().to(device)
track_linear3.load_state_dict(torch.load(filename+'L3.h5',map_location='cpu'))
track_linear4 = Track_LinearNext(mt1+1,mt2+1, 8, 3).double().to(device)
track_linear4.load_state_dict(torch.load(filename+'L4.h5',map_location='cpu'))

track_linear5 = Track_LinearNext(mt1+1,mt2+1, 8, 4).double().to(device)
track_linear5.load_state_dict(torch.load(filename+'L5.h5',map_location='cpu'))

track_linear6 = Track_LinearNext(mt1+1,mt2+1, 8, 5).double().to(device)
track_linear6.load_state_dict(torch.load(filename+'L6.h5',map_location='cpu'))

track_linear7 = Track_LinearNext(mt1+1,mt2+1, 8, 6).double().to(device)
track_linear7.load_state_dict(torch.load(filename+'L7.h5',map_location='cpu'))

track_linear8 = Track_LinearNext(mt1+1,mt2+1, 8, 7).double().to(device)
track_linear8.load_state_dict(torch.load(filename+'L8.h5',map_location='cpu'))


def append_zeroes(length, list_):
    """
    Appends Nones to list to get length of list equal to `length`.
    If list is too long raise AttributeError
    """
    list_=list_.tolist()
    diff_len = length - len(list_)
    if diff_len < 0:
        raise AttributeError('Length error list is too long.')
    list_ = list_ + [-20] * diff_len
    return np.asarray(list_)


def pred_phi1(plev):
    x00 = -9999 * np.ones((len(plev), MAX_TRACKS + 1))
    x01 = -9999 * np.ones((len(plev), MAX_TRACKS + 1))
    x02 = -9999 * np.ones((len(plev), MAX_TRACKS + 1))
    
    for i in range(len(plev)):
        x0 = distphi(plev.iloc[i]['phi0'], plev.phi0.values)
        x1 = distphi(plev.iloc[i]['phi0'], plev.phi1.values)
        x2 = distphi(plev.iloc[i]['phi0'], plev.phi2.values)

        x0.sort()
        x1.sort()
        x2.sort()

        middle_pos = np.argwhere(np.abs(x0) == np.abs(x0).min())
        #print(x0)
        x0 = x0[middle_pos[0,0] - MAX_TRACKS // 2 : middle_pos[0,0] + MAX_TRACKS // 2 + 1]

        #print(x0.shape)
        middle_pos = np.argwhere(np.abs(x1) == np.abs(x1).min())
        x1 = x1[middle_pos[0,0] - MAX_TRACKS // 2 : middle_pos[0,0] + MAX_TRACKS // 2 + 1]
        #rint(x1.shape)
        middle_pos = np.argwhere(np.abs(x2) == np.abs(x2).min())
        x2 = x2[middle_pos[0,0] - MAX_TRACKS // 2 : middle_pos[0,0] + MAX_TRACKS // 2 + 1]
        #rint(x2.shape)
        #print(x0.size)
        if(len(x0) != MAX_TRACKS+1):
            #rint(x0.tolist())
            print('1')
            x00[i] = append_zeroes(MAX_TRACKS+1, x0)
            print(x00[i])
        else:
            x00[i] = x0
        if(len(x1) != MAX_TRACKS+1):
            #rint(x0.tolist())
            print('2')
            x01[i] = append_zeroes(MAX_TRACKS+1, x1)
            print(x01[i])
        else:
            x01[i] = x1
        if(len(x2) != MAX_TRACKS+1):
            #rint(x0.tolist())
            print('3')
            print(x2)
            x02[i] = append_zeroes(MAX_TRACKS+1, x2)
            print(x02[i])
        else:
            x02[i] = x2
        
    
    pred1 = (plev['phi0'].values + track_linear1(torch.from_numpy(x00).to(device), 
                                     torch.from_numpy(x01).to(device), 
                                     torch.from_numpy(x02).to(device)).detach().numpy()[:,0]) % (2 * np.pi)
    plev['phi1p'] = pred1
    
    #print(pred1)
    mat = cdist(plev['phi1p'].values.reshape(-1,1), plev['phi1'].values.reshape(-1,1), metric=distphi2)
    row_ind, col_ind = linear_sum_assignment(mat)    
    plev['phi1p'] = plev.iloc[col_ind]['phi1'].values
    return(plev, col_ind)

def pred_phi_next(plev, n):

    if n == 2:
        track_linear = track_linear2
    if n == 3:
        track_linear = track_linear3
    if n == 4:
        track_linear = track_linear4
    if n == 5:
        track_linear = track_linear5
    if n == 6:
        track_linear = track_linear6
    if n == 7:
        track_linear = track_linear7
    if n == 8:
        track_linear = track_linear8
    
    x00 = -9999 * np.ones((len(plev), MAX_TRACKS + 1))
    x01 = -9999 * np.ones((len(plev), MAX_TRACKS + 1))
    yd = np.zeros((len(plev), n - 1))
    
    for i in range(len(plev)):
        y0 = plev.iloc[i]['phi0']
        for l in range(1, n):
            yd[i, l - 1] = distphi(y0, plev.iloc[i]['phi' + str(l) + 'p'])
            #print(yd)
            y0 = plev.iloc[i]['phi' + str(l) + 'p']
        
        x0 = distphi(plev.iloc[i]['phi' + str(n - 1) + 'p'], plev['phi' + str(n - 1) + 'p'])
        x1 = distphi(plev.iloc[i]['phi' + str(n - 1) + 'p'], plev['phi' + str(n)])

        x0.sort()
        x1.sort()

        middle_pos = np.argwhere(np.abs(x0) == np.abs(x0).min())
        x0 = x0[middle_pos[0,0] - MAX_TRACKS // 2 : middle_pos[0,0] + MAX_TRACKS // 2 + 1]

        middle_pos = np.argwhere(np.abs(x1) == np.abs(x1).min())
        x1 = x1[middle_pos[0,0] - MAX_TRACKS // 2 : middle_pos[0,0] + MAX_TRACKS // 2 + 1]
        
        x00[i] = append_zeroes(MAX_TRACKS+1,x0)
        x01[i] = append_zeroes(MAX_TRACKS+1,x1)        
        

    pred = (plev['phi' + str(n - 1) + 'p'].values + track_linear(torch.from_numpy(x00).to(device), 
                                     torch.from_numpy(x01).to(device), 
                                     torch.from_numpy(yd).to(device)).detach().numpy()[:,0]) % (2 * np.pi)
    #print(pred.shape)
    #print(plev)
    plev['phi' + str(n) + 'p'] = pred
    mat = cdist(plev['phi' + str(n) + 'p'].values.reshape(-1,1), plev['phi' + str(n)].values.reshape(-1,1), metric=distphi2)
    #print(mat.shape)
    row_ind, col_ind = linear_sum_assignment(mat)    
    plev['phi' + str(n) + 'p'] = plev.iloc[col_ind]['phi' + str(n)].values
    return(plev, col_ind)
events = sim_data_test.groupby('event')

predicted_phi = pd.DataFrame(columns=['ev', 'phi0', 'phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7', 'phi8', 'phi1p', 'phi2p', 'phi3p', 'phi4p', 'phi5p', 'phi6p', 'phi7p', 'phi8p'])
predicted_tracks = pd.DataFrame(columns=['ev','phi0', 'phi1p', 'phi2p', 'phi3p', 'phi4p', 'phi5p', 'phi6p', 'phi7p', 'phi8p'])

kk = 0
for name, ev in events:

    pl0 = ev[ev.layer == 0][['cluster_id', 'phi']]; pl0.index = pl0.cluster_id; pl0 = pl0.drop('cluster_id', axis=1);
    pl1 = ev[ev.layer == 1][['cluster_id', 'phi']]; pl1.index = pl1.cluster_id; pl1 = pl1.drop('cluster_id', axis=1);
    pl2 = ev[ev.layer == 2][['cluster_id', 'phi']]; pl2.index = pl2.cluster_id; pl2 = pl2.drop('cluster_id', axis=1);
    pl3 = ev[ev.layer == 3][['cluster_id', 'phi']]; pl3.index = pl3.cluster_id; pl3 = pl3.drop('cluster_id', axis=1);
    pl4 = ev[ev.layer == 4][['cluster_id', 'phi']]; pl4.index = pl4.cluster_id; pl4 = pl4.drop('cluster_id', axis=1);
    pl5 = ev[ev.layer == 5][['cluster_id', 'phi']]; pl5.index = pl5.cluster_id; pl5 = pl5.drop('cluster_id', axis=1);
    pl6 = ev[ev.layer == 6][['cluster_id', 'phi']]; pl6.index = pl6.cluster_id; pl6 = pl6.drop('cluster_id', axis=1);
    pl7 = ev[ev.layer == 7][['cluster_id', 'phi']]; pl7.index = pl7.cluster_id; pl7 = pl7.drop('cluster_id', axis=1);
    pl8 = ev[ev.layer == 8][['cluster_id', 'phi']]; pl8.index = pl8.cluster_id; pl8 = pl8.drop('cluster_id', axis=1);

    plev = pd.concat([pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8], axis=1, join='inner')
    plev.columns = ['phi0', 'phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7', 'phi8']    
    plev['phi1p'] = plev['phi2p'] = plev['phi3p'] = plev['phi4p'] = plev['phi5p'] = plev['phi6p'] = plev['phi7p'] = plev['phi8p'] = 0.
    
    p_tracks = pd.DataFrame(index=range(len(plev)), columns=['ev','phi0', 'phi1p', 'phi2p', 'phi3p', 'phi4p', 'phi5p', 'phi6p', 'phi7p', 'phi8p'])
    p_tracks['ev'] = name
    p_tracks['phi0'] = p_tracks.index

    #print(plev)
    
    plev, p_tracks['phi1p'] = pred_phi1(plev)
    plev, p_tracks['phi2p'] = pred_phi_next(plev, 2)
    plev, p_tracks['phi3p'] = pred_phi_next(plev, 3)
    plev, p_tracks['phi4p'] = pred_phi_next(plev, 4)
    plev, p_tracks['phi5p'] = pred_phi_next(plev, 5)
    plev, p_tracks['phi6p'] = pred_phi_next(plev, 6)
    plev, p_tracks['phi7p'] = pred_phi_next(plev, 7)
    plev, p_tracks['phi8p'] = pred_phi_next(plev, 8)
    
    plev['ev'] = name
    predicted_phi = pd.concat([predicted_phi, plev])
    predicted_tracks_1 = pd.concat([predicted_tracks, p_tracks])
    predicted_tracks = predicted_tracks_1

predicted_tracks.to_csv('predicted_'+str(N)+'_M'+str(M)+'.csv', sep=',', encoding='utf-8', index=False, header = True)
predicted_tracks = pd.read_csv('predicted_'+str(N)+'_M'+str(M)+'.csv', sep=',', encoding='utf-8')
plev.to_csv('plev_'+str(N)+'_M'+str(M)+'.csv', sep=',', encoding='utf-8', index=True, header = True)
plev = pd.read_csv('plev_'+str(N)+'_M'+str(M)+'.csv', sep=',', encoding='utf-8')


import matplotlib.patches as mpatches
import seaborn as sns
events = predicted_tracks.groupby('ev')
SCORE_SUM = 0
NUM_EVENTS = len(events)
EVENT_SCORES = []
for name, group in events:
    predicted_tracks_ev = group
    NUM_TRACKS = len(predicted_tracks_ev)
    SCORE_SUM_EVENT = 0
    checked = []
    for i in range(NUM_TRACKS):
        a = np.array(predicted_tracks_ev.iloc[[i]].values)
        a = a[0,1:10]
        #print(a)
        b = Counter(a)
        C = b.most_common(1)[0]
        #print(C[1])
        if(C[1] >=3 & (C[0] in checked) == False ):
            score_track  = C[1]/9
            #print(C[0])
            checked.append(C[0])
            #print(score_track)
            SCORE_SUM_EVENT = SCORE_SUM_EVENT + score_track
    SCORE_SUM_EVENT = SCORE_SUM_EVENT/NUM_TRACKS
    if(SCORE_SUM_EVENT < 0.90):
        print("Lowest events: %s Score:%8.6f" %(name,SCORE_SUM_EVENT))
    if(SCORE_SUM_EVENT == 1):
        print("Highest events: %s Score:%8.6f" %(name,SCORE_SUM_EVENT))
        
        
    EVENT_SCORES.append(SCORE_SUM_EVENT)
    SCORE_SUM = SCORE_SUM + SCORE_SUM_EVENT        
    #print("_________________________")
    #print("The event: %d score is %".format(float(SCORE_SUM_EVENT)))
    #print("_________________________")
np.savetxt('SCORE_N'+str(N)+'_M'+str(M), EVENT_SCORES, delimiter=',')      
import statistics
stdev = statistics.stdev(EVENT_SCORES)
print(N)
print(M)
print(stdev)
SCORE_SUM = SCORE_SUM/NUM_EVENTS

print(SCORE_SUM)

plt.figure(figsize=(12, 8))
plt.plot(EVENT_SCORES, '-o',c = 'blue')

red_patch = mpatches.Patch(color='red', label='Mean Accuracy', )
blue_patch = mpatches.Patch(color='blue', label= 'Accuracy', )
#blue_dot = plt.plot( "ro", markersize=100)
green_patch = mpatches.Patch(color='green', label='Median Accuracy')

plt.legend([red_patch, green_patch,blue_patch ], ["Mean accuracy", "Median accuracy","Accuracy"])
sns.set_style("whitegrid")
plt.plot(range(100), [np.array(SCORE_SUM).mean()]*100, color = 'red')
plt.plot(range(100), [np.median(EVENT_SCORES)]*100, color = 'green')
plt.xlabel('Event Number')
plt.ylabel('Accuracy')
plt.savefig('accuracy_N'+str(N)+'_M'+str(M)+".png")


filename =  'accuracy'+'_M'+str(M)+".txt"
f = open(filename, "a")
f.write("%s,%s,%8.6f,%8.6f" % (str(M), str(N), SCORE_SUM,stdev ))
f.close()

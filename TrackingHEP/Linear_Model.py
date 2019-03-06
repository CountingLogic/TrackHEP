

###############################################
###############################################
### Model written by : Marco Cristoforetti ####
### Tools added by : Sagar Malhotra        ####
###############################################
###############################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import errno
import argparse
from tqdm import tqdm, tqdm_notebook
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as utils_data
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Runs NN model')
parser.add_argument('--N', type=int, default=10, required=False)
parser.add_argument('--mt0',type=int, default=4,required=False)
parser.add_argument('--mt1',type=int, default=4,required=False)
parser.add_argument('--mt2',type=int, default=4,required=False)
parser.add_argument('--E',type=int, default=3000,required=False)
parser.add_argument('--B',type=int, default=50,required=False)
parser.add_argument('--e',type=int, default=100,required=False)
parser.add_argument('--L',type=int, default=2,required=False)

# N : vertices
# M: maxtracks
# E: epochs
#B: BATCH_SIZE
#E: EVENTS
args = parser.parse_args()
N = args.N
#MAX_TRACKS = args.M
num_epochs = args.E
BATCH_SIZE = args.B
EVENTS = args.e  
mt0 = args.mt0
mt1 = args.mt1
mt2 = args.mt2
LPRED = args.L 

FOLDER = "MODEL_v"+ str(N)+"_e"+str(EVENTS)+"_mt"+"_" +str(mt1)+"_"+str(mt2) + "_B" + str(BATCH_SIZE) + "_E" + str(num_epochs)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder("./"+ FOLDER) 



filename_train ="hits_" + "v"+ str(N) + "_e" + str(EVENTS) + "_" + str(1) + "_new.csv"
filename_test  ="hits_" + "v"+ str(N) + "_e" + str(EVENTS) + "_" + str(2) + "_new.csv"
print(filename_train)
print(filename_test )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Import Data

TRAIN_DATA_PATH = '%s'%filename_train
TEST_DATA_PATH = '%s'%filename_test

# Naming headers for training and testing data 

sim_data_train = pd.read_csv(TRAIN_DATA_PATH, header=None)
print(sim_data_train.head())
sim_data_train.columns = ['event', 'vertex', 'particle', 'layer', 'iphi', 'x', 'y','cluster_id','hit_id','phi']

sim_data_test = pd.read_csv(TEST_DATA_PATH, header=None)
sim_data_test.columns = ['event', 'vertex', 'particle', 'layer', 'iphi', 'x', 'y','cluster_id', 'hit_id', 'phi']

#print(list(sim_data_test.columns.values))

#Creating Cluster ids
#No physics or ML here, just Nomenclature 
#One vertex and one particle make for a unique track_id
#sim_data_train['cluster_id'] = sim_data_train['vertex'] * 100 + sim_data_train['particle']
#sim_data_test['cluster_id'] = sim_data_test['vertex'] * 100 + sim_data_test['particle']

#Definition of the angle distance taking care of sign and periodicity
z2polar = lambda z: [ abs(z[0] + 1j * z[1]), np.angle(z[0] + 1j * z[1]) ]

#distphi is just the implimantation of periodic boundary conditions
'''
if (periodic_x) then
  dx = x(j) - x(i)
  if (dx >   x_size * 0.5) dx = dx - x_size
  if (dx <= -x_size * 0.5) dx = dx + x_size
  https://en.wikipedia.org/wiki/Periodic_boundary_conditions
end if
'''

def distphi(x, y):
    dimensions = 2 * np.pi
    delta = y - x 
    delta = np.where(np.abs(delta) > 0.5 * dimensions, delta - np.sign(delta) * dimensions, delta)
    return delta

#z2polar(x,y) returns (root(x^2 + y^2), arctan(y/x)) , we add pi, to change
#the range from [-pi,pi] to [0,2*pi]
#sim_data_train['phi'] = sim_data_train.apply(lambda x: z2polar(x[['x', 'y']])[1] + np.pi, axis=1)
#sim_data_test['phi'] = sim_data_test.apply(lambda x: z2polar(x[['x', 'y']])[1] + np.pi, axis=1)

MAX_TRACKS = 4 

def gen_dataset(data):
    events = data.groupby('event')
    dataset_l0 = []

    for name, ev in events:
        pl0 = ev[ev['layer'] == 0]
        pl1 = ev[ev['layer'] == 1]
        pl2 = ev[ev['layer'] == 2]

        l0_cluster = pl0['cluster_id']
        
        for track in l0_cluster:
            x = pl0[pl0['cluster_id'] == track]['phi'].values
            y = pl1[pl1['cluster_id'] == track]['phi'].values

            yd = distphi(x, y)
            x0 = distphi(x, pl0.phi.values)
            x1 = distphi(x, pl1.phi.values)
            x2 = distphi(x, pl2.phi.values)

            if yd.size == 0:
                continue
            else:

                len_x0 = x0.shape[0]
                len_x1 = x1.shape[0]
                len_x2 = x2.shape[0]
                
                x0.sort()
                x1.sort()
                x2.sort()
                
                middle_pos = np.argwhere(np.abs(x0) == np.abs(x0).min())
                x0 = x0[middle_pos[0,0] - mt0 // 2 : middle_pos[0,0] + mt0 // 2 + 1]

                middle_pos = np.argwhere(np.abs(x1) == np.abs(x1).min())
                x1 = x1[middle_pos[0,0] - mt1 // 2 : middle_pos[0,0] + mt1 // 2 + 1]

                middle_pos = np.argwhere(np.abs(x2) == np.abs(x2).min())
                x2 = x2[middle_pos[0,0] - mt2 // 2 : middle_pos[0,0] + mt2 // 2 + 1]
                
                dataset_l0.append(dict({'ev': name, 'track': track, 'x': x, 'y': y, 'yd': yd, 'x0': x0, 'x1': x1, 'x2': x2}))    
    return dataset_l0

dataset_l0_train = gen_dataset(sim_data_train)
dataset_l0_test = gen_dataset(sim_data_test)

class trackDataset(utils_data.Dataset):

    def __init__(self, layer_dataset):
        self.layer_dataset = layer_dataset
    
    def __len__(self):
        return self.layer_dataset.shape[0]

    def __getitem__(self, idx):
        sample = {'x0': self.layer_dataset[idx]['x0'], 
                  'x1': self.layer_dataset[idx]['x1'], 
                  'x2': self.layer_dataset[idx]['x2'], 
                  'y' : self.layer_dataset[idx]['y'],
                  'yd': self.layer_dataset[idx]['yd']}
        return sample
#BATCH_SIZE = 50

track_ds_tr = trackDataset(np.array(dataset_l0_train))
data_loader_tr = utils_data.DataLoader(track_ds_tr, batch_size=BATCH_SIZE, shuffle=True)

track_ds_te = trackDataset(np.array(dataset_l0_test))
data_loader_te = utils_data.DataLoader(track_ds_te, batch_size=BATCH_SIZE, shuffle=True)

print(data_loader_tr)

class Track_Linear(nn.Module):
    def __init__(self,mt0, mt1, mt2, lin_h_o):
        super(Track_Linear, self).__init__()

        self.mt1 = mt1
        self.mt2 = mt2
        self.mt0 = mt0

        self.lin_h_o = lin_h_o
        self.lin_h = self.lin_h_o * 2
        self.lin2_h = self.lin_h_o * 2 * 2
        
        #self.len_yd = len_yd
        
        self.linear1a1 = nn.Linear(self.mt0, self.lin_h)
        self.linear2a1 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a1 = nn.Linear(self.lin_h + 1, self.lin_h_o)

        self.linear1a2 = nn.Linear(self.mt1, self.lin_h)
        self.linear2a2 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a2 = nn.Linear(self.lin_h + 1, self.lin_h_o)

        self.linear1a3 = nn.Linear(self.mt2, self.lin_h)
        self.linear2a3 = nn.Linear(self.lin_h, self.lin_h + 1)
        self.linear3a3 = nn.Linear(self.lin_h + 1, self.lin_h_o)

        
        self.linear = nn.Linear(self.lin_h_o * 3, self.lin2_h)        
        self.linear2 = nn.Linear(self.lin2_h, self.lin2_h // 2 + 1)
        self.linear3 = nn.Linear(self.lin2_h // 2 + 1, self.lin_h_o)

        self.lin_last = self.lin_h_o 
        self.linear4 = nn.Linear(self.lin_last, self.lin_last * 2 + 1)
        self.linear5 = nn.Linear(self.lin_last * 2 + 1, self.lin_last // 2  + 1)
        self.linear6 = nn.Linear(self.lin_last // 2 + 1, 1)
             
    def forward(self,x0, x1, x2, yd):
        
        h1 = self.linear1a1(x1)
        h1 = self.linear2a1(h1)
        h1 = self.linear3a1(h1)

        h2 = self.linear1a2(x2)
        h2 = self.linear2a2(h2)
        h2 = self.linear3a2(h2)
        
        h3 = self.linear1a2(x0)
        h3 = self.linear2a2(h3)
        h3 = self.linear3a2(h3)

        h = torch.cat([h1, h2, h3], dim=1)
        x_out = self.linear(h)
        x_out = F.relu(x_out)
        x_out = self.linear2(x_out)
        x_out = F.relu(x_out)
        x_out = self.linear3(x_out)
    
        x_last = x_out
        x_last = self.linear4(x_last)
        x_last = F.relu(x_last)
        x_last = self.linear5(x_last)
        x_last = F.relu(x_last)
        x_last = self.linear6(x_last)
        
        return x_last

def fix_size_input(x):
    x = torch.transpose(x,0,1).contiguous()
    x = x.view(x.shape[0], x.shape[1], 1)
    return x.to(device)

def train_epoch(Xy, model, optimizer, loss_f):

    x0 = Xy ['x0'].to(device)
    x1 = Xy['x1'].to(device)
    x2 = Xy['x2'].to(device)

    yd = Xy['yd'].to(device)
    #ytrue = Xy['y'].to(device)

    model.train()

    optimizer.zero_grad()
    
    outputs = model(x0, x1, x2, yd)

    loss = loss_f(outputs, yd)

    loss.backward()
    optimizer.step()
    return loss 

torch.manual_seed(1237);

#len_yd = LPRED - 1i
track_lstm = Track_Linear(mt0+1,mt1+1,mt2+1, 8 ).double().to(device)
print(track_lstm)

 #main loop training the network

loss_f = nn.L1Loss()

#num_epochs = 3000
history = np.zeros((num_epochs, 2))
lrmax = 0.0001
lr = lrmax
optimizer = torch.optim.Adam(track_lstm.parameters(), lr=lr, weight_decay=1e-6)#, amsgrad=True)
for epoch in range(num_epochs):
    print(epoch)
    print(LPRED)
    loss_mean = 0.
    if epoch > 2000:
        lr = 0.00001
        optimizer = torch.optim.Adam(track_lstm.parameters(), lr=lr, weight_decay=1e-6)#, amsgrad=True)

    for i, batch in enumerate(data_loader_tr):

        loss = train_epoch(batch, track_lstm, optimizer, loss_f)

        loss_mean = (loss_mean * i + loss.item()) / (i + 1)

    track_lstm.eval()
    loss_mean_te = 0.
    for i, batch in enumerate(data_loader_te):

        x0 = batch['x0'].to(device)
        x1 = batch['x1'].to(device)
        x2 = batch['x2'].to(device)
 
        yd = batch['yd'].to(device)
        #dtrue = batch['y'].to(device)

        outputs = track_lstm(x0, x1, x2, yd)
        loss_te = loss_f(outputs, yd)                         
        loss_mean_te = (loss_mean_te * i + loss_te.item()) / (i + 1)

    print('Epoch %d: Loss_tr = %8.6f, Loss_te = %8.6f' % (epoch, loss_mean, loss_mean_te))
    history[epoch] = [loss_mean, loss_mean_te]
    final_test_loss = loss_mean_te
    final_train_loss = loss_mean

# loss_f = nn.L1Loss()
# lowest_loss_mean =1
# suff_epoch = []
# final_test_loss = 0
# final_train_loss = 0
# #num_epochs = 1000
# history = np.zeros((num_epochs, 2))
# lrmax = 0.001
# lr = lrmax
# optimizer = torch.optim.Adam(track_lstm.parameters(), lr=lr, weight_decay=1e-6)#, amsgrad=True)
# for epoch in range(num_epochs):
#     loss_mean = 0.
#     if epoch == 1800:
#         lr = 0.0001
#         optimizer = torch.optim.Adam(track_lstm.parameters(), lr=lr, weight_decay=1e-6)#, amsgrad=True)

#     for i, batch in enumerate(data_loader_tr):

#         x0 = batch['x0'].to(device)
#         x1 = batch['x1'].to(device)
#         x2 = batch['x2'].to(device)

#         track_lstm.train()

#         optimizer.zero_grad()

#         outputs = track_lstm(x0, x1, x2)
#         loss = loss_f(outputs, batch['yd'].to(device))

#         loss.backward()
#         optimizer.step()

#         loss_mean = (loss_mean * i + loss.item()) / (i + 1)

#     track_lstm.eval()
#     loss_mean_te = 0.
#     for i, batch in enumerate(data_loader_te):

#         x0 = batch['x0'].to(device)
#         x1 = batch['x1'].to(device)
#         x2 = batch['x2'].to(device)

#         outputs = track_lstm(x0, x1, x2)
#         loss_te = loss_f(outputs, batch['yd'].to(device))                         
#         loss_mean_te = (loss_mean_te * i + loss_te.item()) / (i + 1)
#     if(epoch%100 == 0):
#         print('Epoch %d: Loss_tr = %8.6f, Loss_te = %8.6f' % (epoch, loss_mean, loss_mean_te))
#     history[epoch] = [loss_mean, loss_mean_te]
#     final_test_loss = loss_mean_te
#     final_train_loss = loss_mean
# print(mt0)
# print(type(mt0))
f= open("history_L_"+ str(LPRED)+".txt","a+")
f.write("The loss for Layer %d mt1 %d mt2 %d is  loss_train = %8.6f, loss_test = %8.6f\r\n" % (LPRED,mt1,mt2,final_train_loss,final_test_loss))
f.close()

f= open("history_table"+ str(LPRED)+".txt","a+")
f.write( "%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,\n"%(mt0,mt1,mt2,final_train_loss,final_test_loss))
f.close()
os.chdir(FOLDER)
    
torch.save(track_lstm.state_dict(), "MODEL_v"+ str(N) + "_e"+str(EVENTS)+"_mt"+str(mt0)+"_"+str(mt1)+"_"+str(mt2) + "_B" + str(BATCH_SIZE) + "_E" + str(num_epochs)+ "_L"+ str(LPRED) + ".h5")
np.savetxt("history_v"+ str(N) + "_e"+str(EVENTS)+ "_mt"+str(mt0)+"_"+str(mt1)+"_"+str(mt2) + "_B" + str(BATCH_SIZE) + "_E" + str(num_epochs) +"_L"+ str(LPRED)+ ".csv", history, delimiter=",")
#np.savetxt("epoch_hits_v1_e100.csv", suff_epoch, delimiter =",")



















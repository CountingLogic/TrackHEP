import numpy as np
import pandas as pd
import argparse
from Simulate import Simulator

sim = Simulator()

parser = argparse.ArgumentParser(description='Simulates particles in 2D')
parser.add_argument('--seed', type=int, default=1, required=False)
parser.add_argument('--e', type=int, default=1, required=False)
parser.add_argument('--N',type=int, default=10,required=False)
parser.add_argument('--sigma_N',type=int, default=0,required=False)

args = parser.parse_args()
seed = args.seed
#N = events
N = args.e
mean_N_vertex  = args.N
sigma_N_vertex = args.sigma_N

##########################################
# costants to add physics of interaction #
########## added by SAGAR MALHOTRA #######
##########################################

# min and low energies
a_ene = 500. #MeV
b_ene = 100000.
# costants for number of tracks
Mmin = 1
a_track = 10.
b_track = 100.
# exponent of the function <M>=pow(M_unif/(F)+pow(a_track,-alpha),1./alpha) , number of tracks per event
alpha = 3.
#The law that relates vertex to number of tracks 
F = 0.5*(pow(a_track,1-alpha)-pow(b_track,1-alpha))
# pile - up & beam parameters and IP parameters
#mean_N_vertex = 1


#sigma_N_vertex =0

k_sigma_d0 = 0.09 #mm
beam_x = 0.5 #mm
beam_y = -0.4 #mm
sigma_beam_x = 0.014 #mm
sigma_beam_y = 0.013 #mm
sigma_vtx = 0.025#mm

##########################################
## information verticies of interaction ##
########## added by SAGAR MALHOTRA #######
##########################################
data = pd.DataFrame({'event': [0], 'vertex':[0], 'particle': [0], 'hit': [
                    0], 'layer': [0], 'iphi': [0], 'x': [0.], 'y': [0.]})
data = data.drop(data.index[[0]])

data_particle = pd.DataFrame({'event': [0], 'vertex':[0], 'particle': [0], 'pt': [
                             0.], 'phi': [0.], 'xVtx': [0.], 'yVtx': [0.]})
data_particle = data_particle.drop(data_particle.index[[0]])

np.random.seed(seed)
event_offset = seed * N

print("Will now produce ", N, " events")

for ievent in range(event_offset, N + event_offset):

    if(ievent % 1 == 0):
        print("processing event : ", ievent)
    event = pd.DataFrame({
        'event': [0], 'vertex':[0], 'particle': [0], 'hit': [0],
        'layer': [0], 'iphi': [0],
        'x': [0.], 'y': [0.]})
    event = event.drop(event.index[[0]])

    event_particle = pd.DataFrame({
        'event': [0], 'vertex':[0], 'particle': [0],
        'pt': [0.], 'phi': [0.], 'xVtx': [0.], 'yVtx': [0.]})
    event_particle = event_particle.drop(event_particle.index[[0]])


    sim.detector.reset()
 

    ##########################################
    ## beam parameters/loop for the pile-up ##
    ########## added by SAGAR MALHOTRA #######
    ##########################################

    beam_position = np.array([beam_x + np.random.normal(0., sigma_beam_x), beam_y + np.random.normal(0., sigma_beam_y)])

    N_vtx = int(np.random.normal(mean_N_vertex,sigma_N_vertex))
    for ivtx in range(N_vtx):
        print("Producing ", ivtx+1,"/",N_vtx," verticies")
        vtx_position = beam_position+[np.random.normal(0.,sigma_vtx),np.random.normal(0.,sigma_vtx)]

        Mult = 0
        while Mult == 0:
            #M = np.random.poisson(nperevent)
            M_unif = np.random.uniform(0,1)
            Mult = pow(M_unif/(F)+pow(a_track,-alpha),1./alpha)
            Mult = int(Mult)
            if Mult < Mmin:
                Mult = Mmin

        for p in range(0, Mult):  # generate M tracks
            print(Mult)
            # pt spectra added by F.Follega
            pt_unif = np.random.uniform(0,1)
            pt = a_ene*np.exp(pt_unif*np.log(b_ene/a_ene))

            phi = np.random.uniform(-np.pi, np.pi)
            momentum = np.array([pt * np.cos(phi), pt * np.sin(phi)])
            charge = 2 * np.random.random_integers(0, 1) - 1

            sigma_d0 = k_sigma_d0/pt # Atlas Data 2016, 13 TeV
            position = vtx_position+[np.random.normal(0., sigma_d0), np.random.normal(0., sigma_d0)]
            xVtx = position[0]
            yVtx = position[1]

            # DR simtrack=sim.propagate(position,velocity, step = 20, id=p)
            simtrack = sim.propagate(position, momentum, charge=charge, p_id=p, vtx_id=ivtx)
            simtrack = pd.concat(
                [pd.DataFrame({'event': [ievent] * len(simtrack.index)}),
                 simtrack],
                axis=1
            )
            event = event.append(simtrack, ignore_index=True)

            event_particle = event_particle.append(pd.concat(
                [pd.DataFrame({'event': [ievent], 'vertex':[ivtx],'particle':[p],
                               'pt':[charge * pt], 'phi':[phi],
                               'xVtx':[xVtx], 'yVtx':[yVtx]})])
            )

    hits = sim.detector.getHits()
    hits = hits.iloc[np.random.permutation(len(hits.index))]
    hits = hits.reset_index(drop=True)
    data_event = pd.concat([
        pd.DataFrame({'event': [ievent] * len(hits.index)}),
        hits],
        axis=1
    )

    data = data.append(data_event, ignore_index=True)
    data_particle = data_particle.append(event_particle, ignore_index=True)

for col in ['event', 'vertex', 'particle', 'hit', 'layer', 'iphi']:
    data[col] = data[col].astype('int32')

for col in ['event','vertex','particle']:
    data_particle[col] = data_particle[col].astype('int32')

data = data.drop(['hit'], axis=1)

# precision could probably be reduced
data.to_csv("hits_" + "v"+ str(mean_N_vertex) + "_e" + str(N) + "_" + str(seed) + ".csv", header=(seed == 0),
            columns=['event','vertex', 'particle', 'layer', 'iphi', 'x', 'y'],
            index=False)
data_particle.to_csv("particles_"+ "v"+ str(mean_N_vertex) + "_e" + str(N) + "_" + str(seed) + ".csv", header=(seed == 0),
    columns=['event','vertex', 'particle', 'pt', 'phi', 'xVtx', 'yVtx'],
    index=False)

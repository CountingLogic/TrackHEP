import numpy as np
import pandas as pd
import math
from transform import rotate, deltaPhiAbs
# TODO : inefficient if small pitch, probably the hit retrieving phase.
# Also do not need both track array and hit


# from yetkin rotate 2D vector in place
def rotate_vector(v, deltaphi):
    c, s = math.cos(deltaphi), math.sin(deltaphi)
    xr = c * v[0] - s * v[1]
    yr = s * v[0] + c * v[1]
    v[0] = xr
    v[1] = yr
    return

# from Thomas Boser : intersections between two circles. FIXME only one
# intersection is enough


def pt_dist(p1, p2):
    """ distance between two points described by a list """
    return math.sqrt(abs((p1[0] - p2[0])**2) + abs((p1[1] - p2[1])**2))


def circ_intersect(v0, v1, r0, r1):
    """ return intersection points of two circles """
    dist = pt_dist(v0, v1)  # calculate distance between
    if dist > (r0 + r1):
        return []  # out of range
    if dist < abs(r0 - r1):
        return []  # circle contained
    if dist == 0:
        return []  # same origin

    a = (r0**2 - r1**2 + dist**2) / (2 * dist)
    h = math.sqrt(r0**2 - a**2)

    v2x = v0[0] + a * (v1[0] - v0[0]) / dist
    v2y = v0[1] + a * (v1[1] - v0[1]) / dist

    x3p = v2x + h * (v1[1] - v0[1]) / dist
    y3p = v2y - h * (v1[0] - v0[0]) / dist
    x3n = v2x - h * (v1[1] - v0[1]) / dist
    y3n = v2y + h * (v1[0] - v0[0]) / dist

    return np.array([[x3p, y3p, 0.], [x3n, y3n, 0.]])


def unit_vector(v):  # normalise in place a vector
    v = v / np.linalg.norm(v)
    return v


def to02pi(x):  # map [-pi,pi] to [0,2pi]
    return (x + 2 * math.pi) % (2 * math.pi)


class Particle(object):

    def __init__(self, x=[], vx=[], charge=0, p_id=0, vtx_id=0, irho=-1, iphi=-1):

        self.history = pd.DataFrame()
        self.position = np.zeros(3)
        self.momentum = np.zeros(3)
        self.charge = charge
        self.p_id = p_id
        self.vtx_id = vtx_id
        self.position[0] = x[0]
        self.position[1] = x[1]
        self.layer = irho  # layer index, -1 if origin
        self.iphi = iphi  # layer index, -1 if origin
        self.momentum[0] = vx[0]
        self.momentum[1] = vx[1]
        self.magmom = np.linalg.norm(self.momentum)
        self.traceMin = 0.1

        self.history = pd.DataFrame({'vertex': [self.vtx_id], 'hit': [0],
                                     'particle': [self.p_id], 'hit': [0],
                                     'layer': irho, 'iphi': iphi,
                                     'x': self.position[0],
                                     'y': self.position[1]
                                     }
                                    )
        self.history = self.history.drop(self.history.index[[0]])
        #        print self.history

        pass

    def update(self, acceleration, time, detector, precision, stephit=1):
        self.momentum += acceleration
        self.position += self.momentum / precision

        deflect = detector.deposit(self.position, self.p_id, self.vtx_id)
        if(np.fabs(deflect) > 0):
            self.momentum = rotate(self.momentum, deflect)
        if((time % stephit == 0) &
           (np.linalg.norm(self.position) > self.traceMin)):
            self.history = self.history.append(pd.DataFrame({
                'vertex': [self.vtx_id],
                'particle': [self.p_id],
                'hit': [time],
                'layer': self.layer, 'iphi': self.iphi,
                'x': [self.position[0]], 'y': [self.position[1]]}),
                ignore_index=True
            )
        pass

    def __str__(self):
        return "Particle id=%s at layer %s" \
               "iphi %s (x,y)=(%s,%s) (px,py)=(%s,%s) ch=%s" % (
                   self.p_id, self.layer, self.iphi,
                   self.position[0], self.position[1],
                   self.momentum[0], self.momentum[1],
                   self.charge)


class Detector(object):
    def __init__(self):
        self.inefficiency = 0  # probability for a hit to not be recorded
        # probability for a track to stop (each layer)
        self.stoppingprobability = 0
        self.Nrho = 9
        self.Npipe = 2
        self.range = 5.
        # FIXME to be divided by P (in MeV)
        self.sigmaMS = 13.6 * math.sqrt(0.02)
        # (https://cds.cern.ch/record/2239573) section 2.4 Table 1 and 4.
        self.cells_r = np.array([39, 85, 155, 213, 271, 405, 562, 762, 1000])
        self.Nphi = []
        for i in range(self.Nrho):
            if i < 5:
                pitch = 0.025  # pitch pixel
            else:
                # pitch strip (divided by sqrt(2) given double layer)
                pitch = 0.050

            self.Nphi += [int(self.cells_r[i] * 2 * math.pi / pitch) + 1]
        # print self.Nphi

        self.cells_phi = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_x = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_y = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.dphi = np.zeros(self.Nrho)
        self.detsize = np.zeros(self.Nrho)

        for irho in range(0, self.Nrho):
            self.dphi[irho] = 2. * np.pi / self.Nphi[irho]
            self.detsize[irho] = self.cells_r[
                irho] * 2. * np.pi / self.Nphi[irho]
            for iphi in range(0, self.Nphi[irho]):
                self.cells_phi[irho, iphi] = 2. * \
                    np.pi * iphi / self.Nphi[irho]
                rho = self.cells_r[irho]
                phi = self.cells_phi[irho, iphi]
                self.cells_x[irho, iphi] = rho * np.cos(phi)
                self.cells_y[irho, iphi] = rho * np.sin(phi)

        self.thickness = 0.02
        self.hit_particle = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.hit_vertex = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_width = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_hit = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.history = pd.DataFrame({'particle': [0], 'hit': [0], 'layer': [
                                    0], 'iphi': [0], 'x': [0], 'y': [0]})
        self.history = self.history.drop(self.history.index[[0]])

    def reset(self):
        self.cells_hit = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.history = pd.DataFrame({'particle': [0], 'hit': [0], 'layer': [
                                    0], 'iphi': [0], 'x': [0.], 'y': [0.]})
        self.history = self.history.drop(self.history.index[[0]])

    def deposit(self, position, particle, vertex):
        deflect = 0.
        for irho in range(0, self.Nrho):
            for iphi in range(0, self.Nphi[irho]):
                if(
                   (np.fabs(np.linalg.norm(position) -
                            self.cells_r[irho]) < self.thickness) &
                   (deltaPhiAbs(np.arctan2(position[1], position[
                    0]), self.cells_phi[irho, iphi]) < self.dphi[irho])
                   ):

                    # think about overlap
                    self.hit_particle[irho, iphi] = particle
                    self.hit_vertex[irho, iphi] = particle
                    self.cells_hit[irho, iphi] = 1
                    # multiple scattering, should divide by particle momentum
                    deflect = np.random.normal(0., self.sigmaMS / 1000)
        return deflect

    def getHits(self):
        ihit = 0
        for irho in range(0, self.Nrho):
            for iphi in range(0, self.Nphi[irho]):
                if(self.cells_hit[irho, iphi] == 1):
                    self.history = self.history.append(
                        pd.DataFrame({'vertex':
                                      self.hit_vertex[irho, iphi],
                                      'particle':
                                      self.hit_particle[irho, iphi],
                                      'hit': [ihit], 'layer': [irho],
                                      'iphi': [iphi],
                                      'x': self.cells_x[irho, iphi],
                                      'y': self.cells_y[irho, iphi]}),
                        ignore_index=True)
                    ihit += 1
        self.history = self.history.sort_values(
            by=['particle', 'layer', 'hit']
        )

        return self.history


class Simulator(object):
    def __init__(self):
        self.p = Particle([0, 0, 0], [0, 0, 0]) # first vector argument is position and the second is the velocity (lot of argument more)
        self.detector = Detector()
        self.precision = 100
        self.inv_bmag = 1. / (0.3*2)  # ATLAS R(mm)=PT(MeV)/0.3*(B(T))
        self.hitid = 0  # hit counter

    def force(self, position, momentum):
        #        g = 0.5
        # acc = -g * position / pow(np.linalg.norm(position),3) # gravitational
        # force
        b = 1. / self.precision
        # This is non relativistic!
        acc = - np.cross(momentum, [0, 0, b])
        print("force called")
        return acc

    def propagate_numeric(self, x=[], v=[], step=1, p_id=0, vtx_id=0):
        #        print "New planet"
        self.p = Particle(x, v, p_id, vtx_id)

        for t in range(0, self.precision):
            acceleration = self.force(self.p.position, self.p.momentum)
            self.p.update(acceleration, t, self.detector,
                          self.precision, stephit=step)
#            if(t % 10 == 0):
#                print t, p.position
        print("propogate_numeric called")
        return self.p.history

    def propagate(self, x=[], v=[], charge=1, irhostart=-1, p_id=0, vtx_id=0):
        print("propogate called")
        debug = False
        self.p = Particle(x, v, int(charge), p_id, vtx_id, irhostart)
        if abs(self.p.charge) != 1:
            print("Detector::propagate abs(charge)!=1 not possible !", charge)
            exit()  # very brutal

        if debug:
            print(self.p)

        for irho in range(self.p.layer + 1, self.detector.Nrho):

            # direct extrapolation to next detector
            # this can certainly be improved to reduce angles calculation

            # coordinates of the center of rotation
            # vector from position to center of rotation
            tocenter = - charge * np.cross(self.p.momentum, [0, 0, self.inv_bmag])
            radius = np.linalg.norm(tocenter)
            if debug:
                print("tocenter=", tocenter, " radius=", radius)

            rotcenter = self.p.position + tocenter
            if debug:
                print("rotcenter=", rotcenter)

            nextrho = self.detector.cells_r[irho]
            nextrhocenter = np.zeros(3)
            # could be done more efficiently
            vintersect = circ_intersect(
                rotcenter, nextrhocenter, radius, nextrho)
            if debug:
                print("nextrho= ", nextrho, " vintersect= ", vintersect)

            if len(vintersect) == 0:
                break
            if self.p.charge == 1:
                newposition = vintersect[1]  # FIXME,  first or second
            else:
                newposition = vintersect[0]  # FIXME,  first or second

            newphipos = math.atan2(newposition[1], newposition[0])
            poschange = newposition - self.p.position
            phichange = math.atan2(poschange[1], poschange[0])

            # rotation of momentum vector
            deltaphimom = 2 * \
                (phichange -
                 math.atan2(self.p.momentum[1], self.p.momentum[0]))
            # insert MS there
            newmomentum = np.copy(self.p.momentum)
            newmagmom = np.linalg.norm(self.p.momentum)
            deflect = np.random.normal(0., self.detector.sigmaMS / newmagmom)

            # now rotate momentum vector
            rotate_vector(newmomentum, deltaphimom + deflect)

            # update particle internal state
            self.p.position = newposition
            self.p.momentum = newmomentum
            self.p.magmom = newmagmom
            self.p.layer = irho

            # now determine which detector element has been hit
            # FIXME there should a Detector function for this
            iphi = int(to02pi(newphipos) / (2 * math.pi) *
                       self.detector.Nphi[irho])
            if debug:
                print("newphipos=", newphipos, "iphi=", iphi, "cell phi",
                      self.detector.cells_phi[irho, iphi])

            self.p.iphi = iphi

            # if inefficient do not record the hit
            rnd = np.random.random()
            if rnd > self.detector.inefficiency:
                self.p.history = self.p.history.append(pd.DataFrame({
                    'vertex': [vtx_id],
                    'particle': [self.p.p_id],
                    'hit': [self.hitid], 'layer': [self.p.layer],
                    'x': [self.p.position[0]], 'y': [self.p.position[1]]}),
                    ignore_index=True)

                # have to think about overlap
                self.detector.hit_particle[irho, iphi] = p_id
                self.detector.hit_vertex[irho, iphi] = vtx_id
                self.detector.cells_hit[irho, iphi] = 1

                self.hitid += 1

            if debug:
                print(self.p)

            # if track stop, stop here
            # rnd = np.random.random()
            # if rnd < self.detector.stoppingprobability:
            #     break

        return self.p.history

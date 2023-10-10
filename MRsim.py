import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

# Class defining a vector in spherical coordinates
class Vector3d:
    def __init__(self,phi_angle_deg,theta_angle_deg,r_magnitude):
        self.phi=phi_angle_deg
        self.theta=theta_angle_deg
        self.r=r_magnitude
        self.phi_rad=self.phi*np.pi/180
    def plot_vector(self,ax,color='k'):
        ax.quiver(0,0,np.cos(self.phi*np.pi/180),np.sin(self.phi*np.pi/180), color=color, scale=1)
        ax.set_rmin(0)
        ax.set_rmax(1)
        ax.set_yticklabels([])
    def __repr__(self):
        return '{},{},{}'.format(self.phi,self.theta,self.r)

# Function to get the angle between two vectors
def get_angle_between(v1: Vector3d, v2: Vector3d):
    return np.pi/180*(v1.phi-v2.phi)

# Class defining a macrospin moment
class Moment:
    def __init__(self, name, initial_vector: Vector3d):
        self.name=name
        self.vector=initial_vector
        
    def calculate_energy(self, exchange_moment: Vector3d):
        E=0
        for term in self.hamiltonian.terms:
            if isinstance(term, Exchange):
                #for now only calculate relative phi angle
                E+=term.magnitude*np.cos(get_angle_between(exchange_moment,self.vector))
            if isinstance(term, Anisotropy):
                E+=term.vector.r*np.cos(2*get_angle_between(term.vector,self.vector))
            if isinstance(term, Zeeman):
                E+=-term.vector.r*np.cos(get_angle_between(term.vector,self.vector))
        return E
    
    def __repr__(self):
        return 'moment {} vector: {}\n'.format(self.name, self.vector)

# Classes for various terms in the Hamiltonian
class Anisotropy:
    def __init__(self,name,vector: Vector3d):
        self.name=name
        self.vector=vector
    def __repr__(self):
        return 'Anisotropy: {} {}'.format(self.name,self.vector.__repr__())
class Zeeman:
    def __init__(self,name,vector: Vector3d):
        self.name=name
        self.vector=vector
    def __repr__(self):
        return 'Zeeman: {} {}'.format(self.name,self.vector.__repr__())
class Exchange:
    def __init__(self,name, magnitude):
        self.name=name
        self.magnitude=magnitude # positive exchange for antiferromagnet
    def __repr__(self):
        return 'Exchange: {}, {}'.format(self.name,self.magnitude)

# Class defining the hamiltonian for a macrospin moment
class Hamiltonian:
    def __init__(self):
        self.terms=[]
    def add_anisotropy(self, anisotropy: Anisotropy):
        self.terms.append(anisotropy)
    def add_Zeeman(self, zeeman: Zeeman):
        self.terms.append(Zeeman)
    def add_exchange(self, exchange: Exchange):
        self.terms.append(Exchange)

    def __repr__(self):
        rep_string='terms in Hamiltonian:\n'
        for term in self.terms:
            rep_string+=term.__repr__()+'\n'
        return rep_string

def minimize_energy(moment_list, hamiltonian: Hamiltonian):
    def energy_function(x):
        E1=0
        E2=0
        for term in hamiltonian.terms:
            #for now only calculate relative phi angle
            if isinstance(term, Exchange):
                E1+=term.magnitude*np.cos(x[0]-x[1])
                E2+=term.magnitude*np.cos(x[1]-x[0])
            if isinstance(term, Anisotropy):
                E1+=term.vector.r*np.cos(2*(term.vector.phi_rad-x[0]))
                E2+=term.vector.r*np.cos(2*(term.vector.phi_rad-x[1]))
            if isinstance(term, Zeeman):
                E1+=-term.vector.r*np.cos(term.vector.phi_rad-x[0])
                E2+=-term.vector.r*np.cos(term.vector.phi_rad-x[1])
        # print(E1)
        # print(E2)
        return E1+E2
    
    bnds=((-np.pi*2,np.pi*2),(-np.pi*2,np.pi*2))
    phi_opt=minimize(energy_function,[moment_list[0].vector.phi_rad,moment_list[1].vector.phi_rad],
                     bounds=bnds, method='Nelder-Mead')
    moment_list[0].vector.phi_rad=phi_opt['x'][0]
    moment_list[0].vector.phi=phi_opt['x'][0]*180/np.pi
    moment_list[1].vector.phi_rad=phi_opt['x'][1]
    moment_list[1].vector.phi=phi_opt['x'][1]*180/np.pi
    return moment_list

def calculate_angular_dependence(angles):
    m1ang=[]
    m2ang=[]
    for angle in angles:   
        m1=Moment('m1',Vector3d(90,90,1))
        m2=Moment('m2',Vector3d(-90,90,1))
        Hext=Vector3d(angle,90,5)
        h=Hamiltonian()
        h.add_anisotropy(Anisotropy('biaxial_anisotropy1',Vector3d(45,90,10e-1)))
        h.add_anisotropy(Anisotropy('biaxial_anisotropy2',Vector3d(135,90,10e-1)))
        h.add_anisotropy(Zeeman('external_field',Hext))
        h.add_anisotropy(Exchange('ex',100))
        minimize_energy([m1,m2],h)
        m1ang.append(m1.vector.phi)
        m2ang.append(m2.vector.phi)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_ylim(0, 1)
    line,=ax.plot([0,angles[0]*np.pi/180],[0,1],'b')
    line1,=ax.plot([0,m1ang[0]*np.pi/180],[0,1],'k')
    line2,=ax.plot([0,m2ang[0]*np.pi/180],[0,1],'k')

    def frame(i):
        print(angles[i])
        line.set_data([0,angles[i]*np.pi/180],[0,1])
        line1.set_data([0,m1ang[i]*np.pi/180],[0,1])
        line2.set_data([0,m2ang[i]*np.pi/180],[0,1])
        
    ax.set_rmin(0)

    ax.set_rmax(1.5)
    ax.set_yticklabels([])
    animation = FuncAnimation(fig, func=frame, frames=range(len(angles)), interval=10)
    plt.show()

calculate_angular_dependence(range(0,360,2))

# m1=Moment('m1',Vector3d(90,90,1))
# m2=Moment('m2',Vector3d(0,90,1))
# Hext=Vector3d(0,90,10)
# h=Hamiltonian()
# h.add_anisotropy(Anisotropy('biaxial_anisotropy1',Vector3d(45,90,10)))
# h.add_anisotropy(Anisotropy('biaxial_anisotropy2',Vector3d(135,90,10)))
# h.add_anisotropy(Zeeman('external_field',Hext))
# h.add_anisotropy(Exchange('ex',100))
# minimize_energy([m1,m2],h)
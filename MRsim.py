import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy import optimize
from matplotlib.animation import FuncAnimation

# Class defining a vector in spherical coordinates
class Vector3d:
    def __init__(self,phi_angle_deg,theta_angle_deg,r_magnitude):
        self.phi=phi_angle_deg
        self.theta=theta_angle_deg
        self.r=r_magnitude
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
    def define_hamiltonian(self, hamiltonian):
        self.hamiltonian=hamiltonian
    def calculate_energy(self):
        self.hamiltonian.calculate_energy(self)
        
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
        return 'moment {} vector: {}\n{}'.format(self.name, self.vector,self.hamiltonian)

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

def perturb_moments(moment_list,T):
    return_list=[]
    for i in range(len(moment_list)):
        moment=moment_list[i]

        initial_vector = moment.vector
        new_vector = Vector3d((initial_vector.phi+np.random.randint(0,359))%360,90,1)
        new_moment=copy.deepcopy(moment)
        new_moment.vector=new_vector

        E_difference=new_moment.calculate_energy(moment_list[(i+1)%(len(moment_list))].vector)-moment.calculate_energy(moment_list[(i+1)%(len(moment_list))].vector)
        #if the cost is negative, flip it, if positive, flip it with some probability
        if E_difference < 0:
            moment=new_moment
        elif np.random.uniform() < np.exp(-E_difference/T):
            moment=new_moment
        return_list.append(moment)
    return return_list

def calculate_angular_dependence(angles):
    m1ang=[]
    m2ang=[]
    for angle in angles:   
        m1=Moment('m1',Vector3d(90,90,1))
        m2=Moment('m2',Vector3d(0,90,1))
        Hext=Vector3d(angle,90,10)
        m1H=Hamiltonian()
        m1H.add_anisotropy(Anisotropy('biaxial_anisotropy1',Vector3d(45,90,10)))
        m1H.add_anisotropy(Anisotropy('biaxial_anisotropy2',Vector3d(135,90,10)))
        m1H.add_anisotropy(Zeeman('external_field',Hext))
        m1H.add_anisotropy(Exchange('m2',100))
        m1.define_hamiltonian(m1H)
        m2H=Hamiltonian()
        m2H.add_anisotropy(Anisotropy('biaxial anisotropy1',Vector3d(45,90,10)))
        m2H.add_anisotropy(Anisotropy('biaxial anisotropy2',Vector3d(135,90,10)))
        m2H.add_anisotropy(Zeeman('external_field',Hext))
        m2H.add_anisotropy(Exchange('m1',100))
        m2.define_hamiltonian(m2H)
        N=1000
        for i in range(0,N):
            [m1,m2]=perturb_moments([m1,m2],1e-6)
        m1ang.append(m1.vector.phi)
        m2ang.append(m2.vector.phi)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_ylim(0, 1)
    line,=ax.plot([0,angles[0]*np.pi/180],[0,1],'b')
    line1,=ax.plot([0,m1ang[0]*np.pi/180],[0,1],'k')
    line2,=ax.plot([0,m2ang[0]*np.pi/180],[0,1],'k')

    def frame(i):
        print(i)
        line.set_data([0,angles[i]*np.pi/180],[0,1])
        line1.set_data([0,m1ang[i]*np.pi/180],[0,1])
        line2.set_data([0,m2ang[i]*np.pi/180],[0,1])
        
    ax.set_rmin(0)

    ax.set_rmax(1.5)
    ax.set_yticklabels([])
    animation = FuncAnimation(fig, func=frame, frames=range(len(angles)), interval=100)
    plt.show()

calculate_angular_dependence(range(0,360,10))
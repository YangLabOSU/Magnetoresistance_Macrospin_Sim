import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Class defining a vector in spherical coordinates
class Vector3d:
    def __init__(self,phi_angle_deg,theta_angle_deg,r_magnitude):
        self.phi=phi_angle_deg
        self.theta=theta_angle_deg
        self.r=r_magnitude
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
    def __init__(self,name, magnitude, moment: Moment):
        self.name=name
        self.magnitude=magnitude # positive exchange for antiferromagnet
        self.moment=moment
    def __repr__(self):
        return 'Exchange: {}, {}, {}'.format(self.name,self.magnitude,self.moment.name)

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
    def calculate_energy(self, moment: Moment):
        E=0
        for term in self.terms:
            if isinstance(term, Exchange):
                #for now only calculate relative phi angle
                E+=term.magnitude*np.cos(get_angle_between(term.moment.vector,moment.vector))
            if isinstance(term, Anisotropy):
                E+=term.vector.r*np.cos(2*get_angle_between(term.vector,moment.vector))
            if isinstance(term, Zeeman):
                E+=term.vector.r*np.cos(get_angle_between(term.vector,moment.vector))
        print(E)

    def __repr__(self):
        rep_string='terms in Hamiltonian:\n'
        for term in self.terms:
            rep_string+=term.__repr__()+'\n'
        return rep_string

def perturb_moments()


m1=Moment('m1',Vector3d(90,90,1))
m2=Moment('m2',Vector3d(-90,90,1))
m1H=Hamiltonian()
m1H.add_anisotropy(Anisotropy('uniaxial_anisotropy',Vector3d(0,90,10)))
m1H.add_anisotropy(Zeeman('external_field',Vector3d(45,90,5)))
m1H.add_anisotropy(Exchange('m2',100,m2))
m1.define_hamiltonian(m1H)
m2H=Hamiltonian()
m2H.add_anisotropy(Anisotropy('uniaxial_anisotropy',Vector3d(0,90,10)))
m2H.add_anisotropy(Zeeman('external_field',Vector3d(45,90,5)))
m2H.add_anisotropy(Exchange('m1',100,m1))
m2.define_hamiltonian(m2H)
m1.calculate_energy()
m2.calculate_energy()
# print(m1)
# print(m2)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.optimize import minimize
import matplotlib.animation as mplanim

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

# Class defining a macrospin moment
class Moment:
    def __init__(self, name, initial_vector: Vector3d):
        self.name=name
        self.vector=initial_vector
    
    def __repr__(self):
        return 'moment {} vector: {}\n'.format(self.name, self.vector)

# Classes for various terms in the free_energy
class Anisotropy:
    def __init__(self,name,symmetry, vector: Vector3d):
        self.name=name
        self.symmetry=symmetry
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
class DMI:
    def __init__(self,name, magnitude):
        self.name=name
        self.magnitude=magnitude
    def __repr__(self):
        return 'DMI: {}, {}'.format(self.name,self.magnitude)

# Class defining the free_energy for a macrospin moment
class Free_Energy:
    def __init__(self, name):
        self.name=name
        self.terms=[]
    def add_anisotropy(self, anisotropy: Anisotropy):
        self.terms.append(anisotropy)
    def add_Zeeman(self, zeeman: Zeeman):
        self.terms.append(zeeman)
    def add_exchange(self, exchange: Exchange):
        self.terms.append(exchange)
    def add_DMI(self, dmi: DMI):
        self.terms.append(dmi)

    def change_term_vector(self, get_name: str, new_vector: Vector3d):
        for term in self.terms:
            if term.name == get_name:
                term.vector=new_vector

    def __repr__(self):
        rep_string='terms in free_energy:\n'
        for term in self.terms:
            rep_string+=term.__repr__()+'\n'
        return rep_string

def minimize_energy(moment_list, free_energy: Free_Energy, angle_guess=0):
    def energy_function(x):
        E1=0
        E2=0
        for term in free_energy.terms:
            #for now only calculate relative phi angle
            if isinstance(term, Exchange):
                E1+=term.magnitude*np.cos(x[0]-x[1])
                E2+=term.magnitude*np.cos(x[1]-x[0])
            if isinstance(term, Anisotropy):
                E1+=-term.vector.r*np.cos(term.symmetry*2*(term.vector.phi_rad-x[0]))
                E2+=-term.vector.r*np.cos(term.symmetry*2*(term.vector.phi_rad-x[1]))
            if isinstance(term, Zeeman):
                E1+=-term.vector.r*np.cos(term.vector.phi_rad-x[0])
                E2+=-term.vector.r*np.cos(term.vector.phi_rad-x[1])
            if isinstance(term, DMI):
                E1+=-term.magnitude*np.sin(x[0]-x[1])
                E2+=-term.magnitude*np.sin(x[0]-x[1])
        # print(E1)
        # print(E2)
        return E1+E2
    
    # phi_opt=minimize(energy_function,[moment_list[0].vector.phi_rad,moment_list[1].vector.phi_rad],
    #                  method='CG')
    
    # minimizing function has trouble at zero angle, so start at field angle + 90
    phi_opt=minimize(energy_function,[angle_guess*np.pi/180+np.pi/2,angle_guess*np.pi/180-np.pi/2],
                     method='CG')
    moment_list[0].vector.phi_rad=phi_opt['x'][0]
    moment_list[0].vector.phi=phi_opt['x'][0]*180/np.pi
    moment_list[1].vector.phi_rad=phi_opt['x'][1]
    moment_list[1].vector.phi=phi_opt['x'][1]*180/np.pi
    return moment_list

# Class that holds the result of an angular field sweep
class Angular_Response:

    def __init__(self, number_of_steps=360, I_direction=0, H_magnitude=20):
        self.H_magnitude=H_magnitude
        self.I_direction=I_direction
        self.spin_polarization_direction_rad=self.I_direction*np.pi/180+np.pi/2
        self.angles=np.linspace(0.01,359.999,number_of_steps)
        self.domain_free_energies=[]
        self.domain_moment1_angles=[]
        self.domain_moment2_angles=[]
        self.Rxx=np.zeros(number_of_steps)
        self.Rxy=np.zeros(number_of_steps)
    
    def add_domain_response(self, free_energy: Free_Energy):
        
        free_energy.add_anisotropy(Zeeman('angular_external_field',Vector3d(0,90,self.H_magnitude)))
        m1ang=[]
        m2ang=[]
        # get the equillibrium positions of m1 and m2 for each field angle
        for angle in self.angles:   
            m1=Moment('m1',Vector3d(90,90,1))
            m2=Moment('m2',Vector3d(-90,90,1))
            Hext=Vector3d(angle,90,self.H_magnitude)
            free_energy.change_term_vector('angular_external_field',Hext)
            minimize_energy([m1,m2],free_energy,angle_guess=angle)
            m1ang.append(m1.vector.phi)
            m2ang.append(m2.vector.phi) 
        m1ang=np.asarray(m1ang)
        m2ang=np.asarray(m2ang)
        Neel_direction_rad=np.pi/180*(m1ang+m2ang)/2+np.pi/2
        canting_angle_deg=(m1ang+m2ang)/2+90-m1ang
        print('Canting angle average: {} deg'.format(np.mean(canting_angle_deg)))
        self.domain_free_energies.append(free_energy)
        self.domain_moment1_angles.append(m1ang)
        self.domain_moment2_angles.append(m2ang)

    def calculate_resistance_contributions(self):
        for moment in self.domain_moment1_angles:
            self.Rxx-=np.cos(2*(moment*np.pi/180-self.spin_polarization_direction_rad))
            self.Rxy-=np.sin(2*(moment*np.pi/180-self.spin_polarization_direction_rad))
        for moment in self.domain_moment2_angles:
            self.Rxx-=np.cos(2*(moment*np.pi/180-self.spin_polarization_direction_rad))
            self.Rxy-=np.sin(2*(moment*np.pi/180-self.spin_polarization_direction_rad))
        
    
def make_parameter_chart(flist: list):
    textlist=[]
    for f in flist:
        text='{}\n'.format(f.name)
        for term in f.terms:
            #for now only calculate relative phi angle
            if isinstance(term, Exchange):
                lt='Exchange ({}): '.format(term.name)
                text+='\n{0:<50}{1:>8}'.format(lt,term.magnitude)
            if isinstance(term, Anisotropy):
                lt='K{} anisotropy ({}) at {} deg: '.format(term.symmetry,term.name,term.vector.phi)
                text+='\n{0:<50}{1:>8}'.format(lt,term.vector.r)
            if isinstance(term, Zeeman):
                lt='Zeeman ({}): '.format(term.name)
                text+='\n{0:<50}{1:>8}'.format(lt,term.vector.r)
            if isinstance(term, DMI):
                lt='DMI ({}): '.format(term.name)
                text+='\n{0:<50}{1:>8}'.format(lt,term.magnitude)
        textlist.append(text)
    return textlist

def plot_angular_SMR(angular_response: Angular_Response, savegif=''):
    angular_response.calculate_resistance_contributions()
    number_of_domains=len(angular_response.domain_free_energies)
    # plot results
    fig = plt.figure(figsize=(6, 6))
    ax2 = plt.subplot(212)
    ax3 = plt.subplot(222)
    ax = plt.subplot(221, polar=True)

    #parameter table
    dlist=make_parameter_chart(angular_response.domain_free_energies)
    for i in range(number_of_domains):
        d=dlist[i]
        # hide axes
        # fig.patch.set_visible(False)
        ax3.axis('off')
        ax3.axis('tight')
        ax3.text(-0.06,0.05-i/30,d, fontsize=5, family='monospace')
    # MR plot
    ax2.plot(angular_response.angles,angular_response.Rxx,
             label='R$_{xx}$')
    ax2.plot(angular_response.angles,angular_response.Rxy,
             label='R$_{xy}$')
    
    pointxx,=ax2.plot(angular_response.angles[0],angular_response.Rxx[0],marker='o', color='b')
    pointxy,=ax2.plot(angular_response.angles[0],angular_response.Rxy[0],marker='o', color='r')
    ax2.legend()
    ax2.set_xlabel(r'$\alpha$ (deg)')

    # moments + field animation
    ax.set_ylim(0, 1)
    line,=ax.plot([0,angular_response.angles[0]*np.pi/180],[0,1],'b')
    line3,=ax.plot([0,angular_response.I_direction*np.pi/180],[0,1],'y')
    xdata1=[]
    xdata2=[]
    for j in range(number_of_domains):
        xdata1+=[0,angular_response.domain_moment1_angles[j][0]*np.pi/180]
        xdata2+=[0,angular_response.domain_moment2_angles[j][0]*np.pi/180]
        
    line1,=ax.plot(xdata1,[0,1]*number_of_domains,'k')
    line2,=ax.plot(xdata2,[0,1]*number_of_domains,'k')

    def frame(i):
        # print(angles[i])
        xdata1=[]
        xdata2=[]
        for j in range(number_of_domains):
            xdata1.append([0,angular_response.domain_moment1_angles[j][i]*np.pi/180])
            xdata2.append([0,angular_response.domain_moment2_angles[j][i]*np.pi/180])
        line.set_data([0,angular_response.angles[i]*np.pi/180],[0,1])
        line1.set_data(xdata1,[0,1]*number_of_domains)
        line2.set_data(xdata2,[0,1]*number_of_domains)
        pointxx.set_data(angular_response.angles[i],angular_response.Rxx[i])
        pointxy.set_data(angular_response.angles[i],angular_response.Rxy[i])

    ax.set_rmin(0)

    ax.set_rmax(1.5)
    ax.set_yticklabels([])
    anim = mplanim.FuncAnimation(fig, func=frame, frames=range(len(angular_response.angles)), interval=10)
    plt.show()

    if savegif != '':
        f=savegif
        writergif = mplanim.PillowWriter(fps=30) 
        anim.save(f, writer=writergif)


savefile='./Fe2O3example.gif'
savefile=''
# Define angular response for alpha scan
a=Angular_Response(I_direction=0,H_magnitude=4.5)
# Add anisotropies and exchange for one structural domain
f1=Free_Energy('a along x domain')
f1.add_anisotropy(Anisotropy('uniaxial_anisotropy',1,Vector3d(0,90,1e-3)))
# f1.add_anisotropy(Anisotropy('biaxial_anisotropy',2,Vector3d(45,90,1e-4)))
# f1.add_anisotropy(Anisotropy('triaxial_anisotropy',3,Vector3d(60,90,2e-5)))
f1.add_anisotropy(Exchange('ex',1000))
# f1.add_anisotropy(DMI('dmi',2.2))
a.add_domain_response(f1)

# Add anisotropies and exchange for one structural domain
f2=Free_Energy('a along y domain')
f2.add_anisotropy(Anisotropy('uniaxial_anisotropy',1,Vector3d(90,90,1e-3)))
f2.add_anisotropy(Exchange('ex',1000))
a.add_domain_response(f2)

plot_angular_SMR(a,savegif=savefile)

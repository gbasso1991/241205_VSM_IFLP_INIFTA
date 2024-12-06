#%% VSM -  Ferrosolidos (ferrotec + laurico)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3

def lineal(x,m,n):
    return m*x+n
#%% 
LB100CP2 = np.loadtxt('LB100_CP2.txt',skiprows=12)
H_A  = LB100CP2[:,0] #Gauss
m_A  = LB100CP2[:,1] #emu
mass_sachet_A = 0.0469 #g
mass_sachet_FF_A = 0.0961 #g
mass_FF_A =  mass_sachet_FF_A-mass_sachet_A #g

LB100FP = np.loadtxt('LB100_FP.txt',skiprows=12)
H_B  = LB100FP[:,0] #Gauss
m_B  = LB100FP[:,1] #emu
mass_sachet_B = 0.0603 #g
mass_sachet_FF_B = 0.1093 #g
mass_FF_B =  mass_sachet_FF_B-mass_sachet_B #g

#normalizo momento por masa de NP
m_A= m_A/mass_FF_A # emu/g
m_B= m_B/mass_FF_B # emu/g


fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(H_A,m_A,'o-',label='100_CP2')
ax.plot(H_B,m_B,'o-',label='100_FP')


ax.legend()
plt.legend(ncol=1)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('100CP2 & 100FP')
# plt.savefig('VSM_NE5X_AD.png',dpi=300)
#%% Resto contribucion dia

H_A1=H_A[np.nonzero(H_A>=5000)]
H_A2=H_A[np.nonzero(H_A<=-5000)]
m_A1=m_A[np.nonzero(H_A>=5000)]
m_A2=m_A[np.nonzero(H_A<=-5000)]

(pend_A1,n_A1), pcov= curve_fit(lineal,H_A1,m_A1)
(pend_A2,n_A2), pcov= curve_fit(lineal,H_A2,m_A2)


chi_mass_A = np.mean([pend_A1,pend_A2])
dia_A1 = lineal(H_A1,chi_mass_A,n_A1)
dia_A2 = lineal(H_A2,chi_mass_A,n_A2)

#%%
fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
# ax.plot(H_A1,m_A1,'o-',label='100_CP2')

# ax.plot(H_A2,m_A2,'o-',label='100_CP2')
ax.plot(H_A,m_A,'o-',label='100_CP2')

ax.plot(H_A1,dia_A1,'-',label='AL diamag 1')
ax.plot(H_A2,dia_A2,'-',label='AL diamag 2')

ax.legend()
plt.legend(ncol=1)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('100CP2 & 100FP')
#%% resto contribucion diamagnetica



#%% fit
vol_FF_A = mass_FF_A*1e-3 #mL  #esto venia del script de Gus
mass_Fe_A = 6.8e-3*vol_FF_A
mass_Fe3O4_A = mass_Fe_A /3/55.85*231.563
a = fit3.session(H_A, m_A,fname='NE5X_A',mass = mass_Fe3O4_A,divbymass=True)
#b = fit3.session(H_A, m_A)
a.setp('N0',3e18)
a.fit()
a.update()
a.plot()
a.print_pars()
#a.save()
#%%
LB100FP = np.loadtxt('D.txt',skiprows=12)
H_D  = LB100FP[:,0] #Gauss
m_D = LB100FP[:,1]  #emu

mass_FF_D = 50.6 #mg
vol_FF_D = mass_FF_D*1e-3 #mL  #esto venia del script de Gus
mass_Fe_D = 6.8e-3*vol_FF_D
mass_Fe3O4_D = mass_Fe_D /3/55.85*231.563

b = fit3.session(H_D, m_D,fname='NE5X_D',mass = mass_Fe3O4_D,divbymass=True)

b.setp('N0',3e18)
b.fit()
b.update()
b.plot()
b.print_pars()
b.save()
#%%
fig,ax=plt.subplots(figsize=(7,4.66),constrained_layout=True)
ax.plot(a.X,a.Y,'o-',label='A')
ax.plot(a.X,a.Yfit,'-',label='A fit')

ax.plot(b.X,b.Y,'o-',label='D')
ax.plot(b.X,b.Yfit,'-',label='D fit')

ax.legend()
# plt.plot(f1,g1,'.-')
# plt.plot(f2,g2,'.-')
# plt.plot(f,g_ajustado,'-',c='tab:red',label=f'$\chi$ = {chi_mass_laurico:.2e} emu/gG')
plt.legend(ncol=2)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu)')
plt.title('NE5X - Antes y Despues de RF')
plt.savefig('VSM_NE5X_AD.png',dpi=300)


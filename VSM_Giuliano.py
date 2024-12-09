#%% VSM -  Ferrosolidos (ferrotec + laurico)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt

def lineal(x,m,n):
    return m*x+n
#%% 
LB100CP2 = np.loadtxt('LB100_CP2.txt',skiprows=12)
H_A  = LB100CP2[:,0] #Gauss
m_A  = LB100CP2[:,1] #emu
mass_sachet_A = 0.0469 #g
mass_sachet_FF_A = 0.0961 #g
mass_FF_A =  mass_sachet_FF_A-mass_sachet_A #g
C_A= 1 #g/L = kg/m³

LB100FP = np.loadtxt('LB100_FP.txt',skiprows=12)
H_B  = LB100FP[:,0] #Gauss
m_B  = LB100FP[:,1] #emu
mass_sachet_B = 0.0603 #g
mass_sachet_FF_B = 0.1093 #g
mass_FF_B =  mass_sachet_FF_B-mass_sachet_B #g
C_B= 0.85 #g/L = kg/m³

#normalizo momento por masa de NP
m_A = m_A/mass_FF_A # emu/g
m_B = m_B/mass_FF_B # emu/g


fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(H_A,m_A,'.-',label='100_CP2')
ax.plot(H_B,m_B,'.-',label='100_FP')
ax.legend()
plt.legend(ncol=1)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('Señales originales\n100CP2 & 100FP')
# plt.savefig('VSM_NE5X_AD.png',dpi=300)

#%% uso herramienta para ciclo anhisteretico

H_anhist_A,m_anhist_A = mt.anhysteretic(H_A,m_A)

H_anhist_B,m_anhist_B = mt.anhysteretic(H_B,m_B)

fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(H_anhist_A,m_anhist_A,'.-',label='100_CP2')
ax.plot(H_anhist_B,m_anhist_B,'.-',label='100_FP')
ax.legend()
plt.legend(ncol=1)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('Señales anhistereticas\n100CP2 & 100FP')


# #%% Testeo fit3
# qq=[]
# qq.append(fit3.session (H_anhist_A,m_FF_anhist_A, fname='test_A'))
# qq[0].fix('sig0')
# qq[0].fix('mu0')
# qq[0].free('dc')
# qq[0].fit()
# qq[0].set_yE_as('sep')
# qq[0].fit()
# qq[0].update()
# #qq[0].save()
# qq[0].print_pars()

#%% Calculo contribucion diamag

H_ah_A1=H_anhist_A[np.nonzero(H_anhist_A>=10000)]
H_ah_A2=H_anhist_A[np.nonzero(H_anhist_A<=-10000)]
m_ah_A1=m_FF_anhist_A[np.nonzero(H_anhist_A>=10000)]
m_ah_A2=m_FF_anhist_A[np.nonzero(H_anhist_A<=-10000)]

(pend_A1,n_A1), pcov= curve_fit(lineal,H_ah_A1,m_ah_A1)
(pend_A2,n_A2), pcov= curve_fit(lineal,H_ah_A2,m_ah_A2)

chi_mass_A = np.mean([pend_A1,pend_A2])
print(f'Susceptibilidad masica A (100CP2) = {chi_mass_A}')
dia_A1 = lineal(H_ah_A1,chi_mass_A,n_A1)
dia_A2 = lineal(H_ah_A2,chi_mass_A,n_A2)
#%%

H_ah_B1=H_anhist_B[np.nonzero(H_anhist_B>=10000)]
H_ah_B2=H_anhist_B[np.nonzero(H_anhist_B<=-10000)]
m_ah_B1=m_FF_anhist_B[np.nonzero(H_anhist_B>=10000)]
m_ah_B2=m_FF_anhist_B[np.nonzero(H_anhist_B<=-10000)]

(pend_B1,n_B1), pcov= curve_fit(lineal,H_ah_B1,m_ah_B1)
(pend_B2,n_B2), pcov= curve_fit(lineal,H_ah_B2,m_ah_B2)

chi_mass_B = np.mean([pend_B1,pend_B2])
print(f'Susceptibilidad masica B (100FP) = {chi_mass_B}')
dia_B1 = lineal(H_ah_B1,chi_mass_B,n_B1)
dia_B2 = lineal(H_ah_B2,chi_mass_B,n_B2)

#%%

fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(H_anhist_A,m_anhist_A,'.-',label='100_CP2')
ax.plot(H_ah_A1,dia_A1,'-',label='AL Adiamag 1')
ax.plot(H_ah_A2,dia_A2,'-',label='AL A diamag 2')

ax.plot(H_anhist_B,m_anhist_B,'.-',label='100_FP')
ax.plot(H_ah_B1,dia_B1,'-',label='AL B diamag 1')
ax.plot(H_ah_B2,dia_B2,'-',label='AL B diamag 2')

ax.legend()
plt.legend(ncol=2)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('100CP2 & 100FP')
plt.savefig('VSM_c_diamag.png',dpi=300)

#%% resto contribucion diamagnetica
diamag_A= lineal(H_anhist_A,chi_mass_A,0)
m_A_sin_diamag = m_anhist_A - diamag_A 

diamag_B= lineal(H_anhist_B,chi_mass_B,0)
m_B_sin_diamag = m_anhist_B - diamag_B 


fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(H_anhist_A,m_anhist_A,'.-',label='100_CP2')
ax.plot(H_anhist_A,m_A_sin_diamag,'.-',label='100_CP2')

ax.plot(H_anhist_B,m_anhist_B,'.-',label='100_FP')
ax.plot(H_anhist_B,m_B_sin_diamag,'.-',label='100_FP')
ax.legend()
plt.legend(ncol=2)
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('100CP2 & 100FP')
plt.savefig('VSM_s_diamag_cgs.png',dpi=300)

#%% Realizo fits en ciclos sin contribucion diamag
fit_A = fit3.session(H_anhist_A,m_A_sin_diamag,fname='100_CP2',divbymass=False)
fit_A.fix('sig0')
fit_A.fix('mu0')
fit_A.free('dc')
fit_A.fit()
fit_A.update()
fit_A.free('sig0')
fit_A.free('mu0')
fit_A.set_yE_as('sep')
fit_A.fit()
fit_A.update()
fit_A.save()
fit_A.print_pars()
H_A_fit = fit_A.X
m_A_fit = fit_A.Y

fit_B = fit3.session(H_anhist_B,m_B_sin_diamag,fname='100_CP2',divbymass=False)
fit_B.fix('sig0')
fit_B.fix('mu0')
fit_B.free('dc')
fit_B.fit()
fit_B.update()
fit_B.free('sig0')
fit_B.free('mu0')
fit_B.set_yE_as('sep')
fit_B.fit()
fit_B.update()
fit_B.save()
fit_B.print_pars()
H_B_fit = fit_B.X
m_B_fit = fit_B.Y
#%%
fig,(ax1,ax2)=plt.subplots(nrows=2,figsize=(8,7),sharex=True,constrained_layout=True)
ax1.plot(H_anhist_A,m_A_sin_diamag,'.-',label='100_CP2')
ax1.plot(H_A_fit,m_A_fit,'-',label='fit')

ax2.plot(H_anhist_B,m_B_sin_diamag,'.-',label='100_FP')
ax2.plot(H_B_fit,m_B_fit,'-',label='fit')

for a in [ax1,ax2]:
    a.legend(ncol=2)
    a.grid()
    a.set_ylabel('m (emu/g)')
ax2.set_xlabel('H (G)')
ax1.set_title('100CP2',loc='left')
ax2.set_title('100FP',loc='left')

#%% Paso a unidades de SI , normalizo por Concentracion y comparo ciclos
H_A_fit*=1e3/(4*np.pi) # Oe a A/m
H_B_fit*=1e3/(4*np.pi) # Oe a A/m


#%% Paso a unidades del SI y normalizo por concentracion

# H_A*=1e3/(4*np.pi) # Oe a A/m
# H_B*=1e3/(4*np.pi) # Oe a A/m

# M_A = m_A_sin_diamag/C_A
# M_B = m_B_sin_diamag/C_B

# fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
# ax.plot(H_A,M_A,'o-',label='100_CP2')

# ax.plot(H_B,M_B,'o-',label='100_FP')

# ax.legend()
# plt.legend(ncol=2)
# plt.grid()
# plt.xlabel('H (A/m)')
# plt.ylabel('M (A/m)')
# plt.title('100CP2 & 100FP')
# plt.savefig('VSM_s_diamag_SI.png',dpi=300)
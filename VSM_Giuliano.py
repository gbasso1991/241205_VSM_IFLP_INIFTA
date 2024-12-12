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
        
#%% Leo Archivos 
### 100
# LB100_CP2
LB100CP2 = np.loadtxt('LB100_CP2.txt', skiprows=12)
H_A = LB100CP2[:, 0]  # Gauss
m_A = LB100CP2[:, 1]  # emu
mass_sachet_A = 0.0469  # g
mass_sachet_FF_A = 0.0961  # g
mass_FF_A = mass_sachet_FF_A - mass_sachet_A  # g
C_A = 1  # g/L = kg/m³
#%%
# LB100_FP
LB100FP = np.loadtxt('LB100_FP.txt', skiprows=12)
H_B = LB100FP[:, 0]  # Gauss
m_B = LB100FP[:, 1]  # emu
mass_sachet_B = 0.0603  # g
mass_sachet_FF_B = 0.1093  # g
mass_FF_B = mass_sachet_FF_B - mass_sachet_B  # g
C_B = 0.85  # g/L = kg/m³

# LB100_OH
LB100OH = np.loadtxt('LB100OH.txt', skiprows=12)
H_C = LB100OH[:, 0]  # Gauss
m_C = LB100OH[:, 1]  # emu
mass_sachet_C = 0.0608  # g
mass_sachet_FF_C = 0.1123  # g
mass_FF_C = mass_sachet_FF_C - mass_sachet_C  # g
C_C = 1  # g/L = kg/m³

# LB100_P
LB100P = np.loadtxt('LB100P.txt', skiprows=12)
H_D = LB100P[:, 0]  # Gauss
m_D = LB100P[:, 1]  # emu
mass_sachet_D = 0.0574  # g
mass_sachet_FF_D = 0.1040  # g
mass_FF_D = mass_sachet_FF_D - mass_sachet_D  # g
C_D = 0.79  # g/L = kg/m³
#%%
# LB97_CP2
LB97CP2 = np.loadtxt('LB97CP2.txt', skiprows=12)
H_E = LB97CP2[:, 0]  # Gauss
m_E = LB97CP2[:, 1]  # emu
mass_sachet_E = 0.0642  # g
mass_sachet_FF_E = 0.1152  # g
mass_FF_E = mass_sachet_FF_E - mass_sachet_E  # g
C_E = 1  # g/L = kg/m³

# LB97_FP
LB97FP = np.loadtxt('LB97FP1.txt', skiprows=12)
H_F = LB97FP[:, 0]  # Gauss
m_F = LB97FP[:, 1]  # emu
mass_sachet_F = 0.0645  # g
mass_sachet_FF_F = 0.1062  # g
mass_FF_F = mass_sachet_FF_F - mass_sachet_F  # g
C_F = 0.17  # g/L = kg/m³

# LB97_OH
LB97OH = np.loadtxt('LB97OH.txt', skiprows=12)
H_G = LB97OH[:, 0]  # Gauss
m_G = LB97OH[:, 1]  # emu
mass_sachet_G = 0.0684  # g
mass_sachet_FF_G = 0.1178  # g
mass_FF_G = mass_sachet_FF_G - mass_sachet_G  # g
C_G = 1  # g/L = kg/m³

# LB97_P
LB97P = np.loadtxt('LB97P.txt', skiprows=12)
H_H = LB97P[:, 0]  # Gauss
m_H = LB97P[:, 1]  # emu
mass_sachet_H = 0.0631  # g
mass_sachet_FF_H = 0.1133  # g
mass_FF_H = mass_sachet_FF_H - mass_sachet_H  # g
C_H = 0.95  # g/L = kg/m³


# Normalizo momento por masa de NP
m_A /= mass_FF_A  # emu/g
m_B /= mass_FF_B  # emu/g
m_C /= mass_FF_C  # emu/g
m_D /= mass_FF_D  # emu/g
m_E /= mass_FF_E  # emu/g
m_F /= mass_FF_F  # emu/g
m_G /= mass_FF_G  # emu/g
m_H /= mass_FF_H  # emu/g

# Graficar
fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(12, 8), constrained_layout=True)
ax1.plot(H_A, m_A, '.-', label='100_CP2')
ax1.plot(H_B, m_B, '.-', label='100_FP')
ax1.plot(H_C, m_C, '.-', label='100_OH')
ax1.plot(H_D, m_D, '.-', label='100_P')
ax2.plot(H_E, m_E, '.-', label='97_CP2')
ax2.plot(H_F, m_F, '.-', label='97_FP')
ax2.plot(H_G, m_G, '.-', label='97_OH')
ax2.plot(H_H, m_H, '.-', label='97_P')

for a in [ax1,ax2]:
    a.legend(ncol=2)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.suptitle('Señales originales\n100_CP2, 100_FP, 100_OH, 100_P, 97_CP2, 97_FP, 97_OH, 97_P, 97_FP1')
plt.show()

#%% uso herramienta para ciclo anhisteretico
# Generar señales anhisteréticas
H_anhist_A, m_anhist_A = mt.anhysteretic(H_A, m_A)
H_anhist_B, m_anhist_B = mt.anhysteretic(H_B, m_B)
H_anhist_C, m_anhist_C = mt.anhysteretic(H_C, m_C)
H_anhist_D, m_anhist_D = mt.anhysteretic(H_D, m_D)
H_anhist_E, m_anhist_E = mt.anhysteretic(H_E, m_E)
H_anhist_F, m_anhist_F = mt.anhysteretic(H_F, m_F)
H_anhist_G, m_anhist_G = mt.anhysteretic(H_G, m_G)
H_anhist_H, m_anhist_H = mt.anhysteretic(H_H, m_H)

# Graficar señales anhisteréticas
# fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
# ax.plot(H_anhist_A, m_anhist_A, '.-', label='100_CP2')
# ax.plot(H_anhist_B, m_anhist_B, '.-', label='100_FP')
# ax.plot(H_anhist_C, m_anhist_C, '.-', label='100_OH')
# ax.plot(H_anhist_D, m_anhist_D, '.-', label='100_P')
# ax.plot(H_anhist_E, m_anhist_E, '.-', label='97_CP2')
# ax.plot(H_anhist_F, m_anhist_F, '.-', label='97_FP')
# ax.plot(H_anhist_G, m_anhist_G, '.-', label='97_OH')
# ax.plot(H_anhist_H, m_anhist_H, '.-', label='97_P')

# ax.legend()
# plt.grid()
# plt.xlabel('H (G)')
# plt.ylabel('m (emu/g)')
# plt.title('Señales anhisteréticas\n100_CP2, 100_FP, 100_OH, 100_P, 97_CP2, 97_FP, 97_OH, 97_P, 97_FP1')
# plt.show()

#%% Realizo fits en ciclos CON contribucion diamag# Ajustes para LB100_CP2
# Ajustes para LB100_CP2
fit_A = fit3.session(H_anhist_A, m_anhist_A, fname='100_CP2', divbymass=False)
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
m_A_sin_diamag = m_anhist_A - lineal(H_anhist_A, fit_A.params['C'].value, fit_A.params['dc'].value)

# Ajustes para LB100_FP
fit_B = fit3.session(H_anhist_B, m_anhist_B, fname='100_FP', divbymass=False)
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
m_B_sin_diamag = m_anhist_B - lineal(H_anhist_B, fit_B.params['C'].value, fit_B.params['dc'].value)

# Ajustes para LB100_OH
fit_C = fit3.session(H_anhist_C, m_anhist_C, fname='100_OH', divbymass=False)
fit_C.fix('sig0')
fit_C.fix('mu0')
fit_C.free('dc')
fit_C.fit()
fit_C.update()
fit_C.free('sig0')
fit_C.free('mu0')
fit_C.set_yE_as('sep')
fit_C.fit()
fit_C.update()
fit_C.save()
fit_C.print_pars()
H_C_fit = fit_C.X
m_C_fit = fit_C.Y
m_C_sin_diamag = m_anhist_C - lineal(H_anhist_C, fit_C.params['C'].value, fit_C.params['dc'].value)

# Ajustes para LB100_P
fit_D = fit3.session(H_anhist_D, m_anhist_D, fname='100_P', divbymass=False)
fit_D.fix('sig0')
fit_D.fix('mu0')
fit_D.free('dc')
fit_D.fit()
fit_D.update()
fit_D.free('sig0')
fit_D.free('mu0')
fit_D.set_yE_as('sep')
fit_D.fit()
fit_D.update()
fit_D.save()
fit_D.print_pars()
H_D_fit = fit_D.X
m_D_fit = fit_D.Y
m_D_sin_diamag = m_anhist_D - lineal(H_anhist_D, fit_D.params['C'].value, fit_D.params['dc'].value)

# Ajustes para LB97_CP2
fit_E = fit3.session(H_anhist_E, m_anhist_E, fname='97_CP2', divbymass=False)
fit_E.fix('sig0')
fit_E.fix('mu0')
fit_E.free('dc')
fit_E.fit()
fit_E.update()
fit_E.free('sig0')
fit_E.free('mu0')
fit_E.set_yE_as('sep')
fit_E.fit()
fit_E.update()
fit_E.save()
fit_E.print_pars()
H_E_fit = fit_E.X
m_E_fit = fit_E.Y
m_E_sin_diamag = m_anhist_E - lineal(H_anhist_E, fit_E.params['C'].value, fit_E.params['dc'].value)

# Ajustes para LB97_FP
fit_F = fit3.session(H_anhist_F, m_anhist_F, fname='97_FP', divbymass=False)
fit_F.fix('sig0')
fit_F.fix('mu0')
fit_F.free('dc')
fit_F.fit()
fit_F.update()
fit_F.free('sig0')
fit_F.free('mu0')
fit_F.set_yE_as('sep')
fit_F.fit()
fit_F.update()
fit_F.save()
fit_F.print_pars()
H_F_fit = fit_F.X
m_F_fit = fit_F.Y
m_F_sin_diamag = m_anhist_F - lineal(H_anhist_F, fit_F.params['C'].value, fit_F.params['dc'].value)

# Ajustes para LB97_OH
fit_G = fit3.session(H_anhist_G, m_anhist_G, fname='97_OH', divbymass=False)
fit_G.fix('sig0')
fit_G.fix('mu0')
fit_G.free('dc')
fit_G.fit()
fit_G.update()
fit_G.free('sig0')
fit_G.free('mu0')
fit_G.set_yE_as('sep')
fit_G.fit()
fit_G.update()
fit_G.save()
fit_G.print_pars()
H_G_fit = fit_G.X
m_G_fit = fit_G.Y
m_G_sin_diamag = m_anhist_G - lineal(H_anhist_G, fit_G.params['C'].value, fit_G.params['dc'].value)

# Ajustes para LB97_P
fit_H = fit3.session(H_anhist_H, m_anhist_H, fname='97_P', divbymass=False)
fit_H.fix('sig0')
fit_H.fix('mu0')
fit_H.free('dc')
fit_H.fit()
fit_H.update()
fit_H.free('sig0')
fit_H.free('mu0')
fit_H.set_yE_as('sep')
fit_H.fit()
fit_H.update()
fit_H.save()
fit_H.print_pars()
H_H_fit = fit_H.X
m_H_fit = fit_H.Y
m_H_sin_diamag = m_anhist_H - lineal(H_anhist_H, fit_H.params['C'].value, fit_H.params['dc'].value)

# Ajustes para LB97_FP1

# Graficar resultados eliminando comportamiento diamagnético
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, constrained_layout=True)

# Eje superior: Datos de "100"
ax1.plot(H_anhist_A, m_A_sin_diamag, '.-', label='100_CP2')
ax1.plot(H_anhist_B, m_B_sin_diamag, '.-', label='100_FP')
ax1.plot(H_anhist_C, m_C_sin_diamag, '.-', label='100_OH')
ax1.plot(H_anhist_D, m_D_sin_diamag, '.-', label='100_P')
ax1.legend(ncol=2)
ax1.grid()
ax1.set_ylabel('m (emu/g)')
ax1.set_title('Datos sin diamagnético (100)', loc='left')

# Eje inferior: Datos de "97"
ax2.plot(H_anhist_E, m_E_sin_diamag, '.-', label='97_CP2')
ax2.plot(H_anhist_F, m_F_sin_diamag, '.-', label='97_FP')
ax2.plot(H_anhist_G, m_G_sin_diamag, '.-', label='97_OH')
ax2.plot(H_anhist_H, m_H_sin_diamag, '.-', label='97_P')
ax2.legend(ncol=2)
ax2.grid()
ax2.set_ylabel('m (emu/g)')
ax2.set_xlabel('H (G)')
ax2.set_title('Datos sin diamagnético (97)', loc='left')

plt.show()


#%% #%% finalmente multiplico a la señal de mom_magnetico/masa_FF sin diamagnetismo por la densidad del FF (H20) y divido por concentracion
rho_FF= 1000# g/L
m_A_norm=m_A_sin_diamag*rho_FF/C_A # Am²/kg
m_B_norm=m_B_sin_diamag*rho_FF/C_B # Am²/kg
m_C_norm=m_C_sin_diamag*rho_FF/C_C # Am²/kg
m_D_norm=m_D_sin_diamag*rho_FF/C_D # Am²/kg
m_E_norm=m_E_sin_diamag*rho_FF/C_E # Am²/kg
m_F_norm=m_F_sin_diamag*rho_FF/C_F # Am²/kg
m_G_norm=m_G_sin_diamag*rho_FF/C_G # Am²/kg
m_H_norm=m_H_sin_diamag*rho_FF/C_H # Am²/kg

H_A_fit*=1e3/(4*np.pi) # G a A/m
H_B_fit*=1e3/(4*np.pi) # G a A/m
H_C_fit*=1e3/(4*np.pi) # G a A/m
H_D_fit*=1e3/(4*np.pi) # G a A/m
H_E_fit*=1e3/(4*np.pi) # G a A/m
H_F_fit*=1e3/(4*np.pi) # G a A/m
H_G_fit*=1e3/(4*np.pi) # G a A/m
H_H_fit*=1e3/(4*np.pi) # G a A/m
#%% 
# Graficar resultados eliminando comportamiento diamagnético
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, constrained_layout=True)

ax1.set_title('mom magnetico por masa de NP - 100', loc='left')
ax1.plot(H_A_fit, m_A_norm, '.-', label='100_CP2')
ax1.plot(H_B_fit, m_B_norm, '.-', label='100_FP')
ax1.plot(H_C_fit, m_C_norm, '.-', label='100_OH')
ax1.plot(H_D_fit, m_D_norm, '.-', label='100_P')

# Eje inferior: Datos de "97"
ax2.set_title('mom magnetico por masa de NP - 97', loc='left')
ax2.plot(H_E_fit, m_E_norm, '.-', label='97_CP2')
ax2.plot(H_F_fit, m_F_norm, '.-', label='97_FP')
ax2.plot(H_G_fit, m_G_norm, '.-', label='97_OH')
ax2.plot(H_H_fit, m_H_norm, '.-', label='97_P')
ax2.set_xlabel('H (A/m)')

for a in [ax1,ax2]:
    a.legend(ncol=2)
    a.grid()
    a.set_ylabel('m (Am²/kg)')
plt.savefig('Mom_magnetico_por_masa_NPM.png',dpi=400)
plt.show()
#%% Extraigo resultados de los fits

# sus_diamag=[]
# offset=[]
# ms_lin=[]
# mag_sat=[]
# magsat2=[]
# mean_mu_mu=[]
# mean_mu_mu2=[]
# mu0_fit2=[]
# sig0_fit2=[]


# fits= [fit_A,fit_B,fit_C,fit_D,fit_E,fit_F,fit_G,fit_H]

# for fit in fits:
#     sus_diamag.append(fit.params['C'].value)
#     mag_sat.append()




#%% Clase Muestra
class Muestra():
    'Archivo VSM'
    def __init__(self, nombre_archivo, masa_sachet, masa_sachet_NP, concentracion_MNP, err_concentracion_MNP):
        self.nombre_archivo = nombre_archivo
        
        
        self.masa_sachet = masa_sachet
        self.masa_sachet_NP = masa_sachet_NP
        self.concentracion_MNP = concentracion_MNP
        self.err_concentracion_MNP = err_concentracion_MNP

muestras = []

#%% Agregar muestras manualmente (sin vectores previos)
# muestras.append(Muestra('LB100_CP2.txt',0.0469,0.0961,1,0.01)) 
# muestras.append(Muestra('LB100_FP.txt',0.0603, 0.1093,0.85,0.01))         
# muestras.append(Muestra('LB100_OH.txt',0.0608,0.1123,1,0.01)) 
# muestras.append(Muestra('LB100_P.txt',0.0574, 0.1040,0.79,0.01))           

muestras.append(Muestra('LB97_CP2.txt',0.0642,0.1152,1,0.01)) 
muestras.append(Muestra('LB97_FP1.txt',0.0645, 0.1062,0.85,0.01))         
# muestras.append(Muestra('LB97_OH.txt',0.0684,0.1178,1,0.01)) 
# muestras.append(Muestra('LB97_P.txt',0.0631, 0.1133,0.79,0.01))  

nombre_archivo  = [muestra.nombre_archivo for muestra in muestras]
masa_sachet = np.array([muestra.masa_sachet for muestra in muestras])
masa_sachet_NP = np.array([muestra.masa_sachet_NP for muestra in muestras])
concentracion_MNP = np.array([muestra.concentracion_MNP for muestra in muestras])
errores_concentracion = np.array([muestra.err_concentracion_MNP for muestra in muestras])
masa_FF = masa_sachet_NP-masa_sachet

sus_diamag=[]
offset=[]
ms_lin=[]
mag_sat=[]
magsat2=[]
mean_mu_mu=[]
meanmu_mu2=[]
mu0_fit2=[]
sig0_fit2=[]
# %
#Iniciamnos sesión de ajuste (aun no se ajusta) y graficamos curvas originales normalizadas

fit_sessions=[]
for k in range(len(nombre_archivo)):
    #Se lee el archivo
    archivo=nombre_archivo[k]
    data = np.loadtxt (archivo, skiprows=12)
    (campo,momento) = (data[:,0],data[:,1])
    
    #Normalizamos por masa de FF
    magnetizacion_FF=momento/masa_FF[k] # /concentracion_MNP[k]*1000 #(emu/g)
           
    #Armamos la curva anhisteretica
    campo_anhist,magnetizacion_FF_anhist = mt.anhysteretic(campo,magnetizacion_FF)
    
    # #Se inicia la sesión de ajuste con la curva anhisterética
    fit_sessions.append(fit3.session (campo_anhist,magnetizacion_FF_anhist, fname='anhi'))
    fit=fit_sessions[k]
    fit.label=archivo[:-4]

    # plt.plot(campo, magnetizacion_FF,'ro',label=fit_sessions[k].label)
    # plt.xlabel('Campo (G)')
    # plt.ylabel('Magnetizacion (emu/g)')
    # plt.title('Normalizado por masa de FF')
    # plt.legend()
    # plt.savefig('Ciclos_normalizados_por_masa_de_FF')
    # plt.show()
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()

    H_fit = fit.X
    m_fit = fit.Y
    m_sin_diamag = magnetizacion_FF_anhist - lineal(campo_anhist, fit.params['C'].value, fit.params['dc'].value)
    m_sin_diamag_norm = m_sin_diamag/concentracion_MNP[k]*1000
    
    plt.figure(112)
    plt.plot(campo_anhist,m_sin_diamag_norm,'o-',label=fit.label)
    plt.xlabel('Campo (Oe)')
    plt.ylabel('Momento (emu/g)')
    plt.title('Magnetización sin señal diamagnétcia normalizado por masa de MNP')
    plt.legend()
    plt.show()
    
    plt.figure(114)
    plt.plot(campo_anhist, (magnetizacion_FF_anhist-fit.params['C'].value*campo_anhist-fit.params['dc']),label=fit.label)
    plt.xlabel('Campo (G)')
    plt.ylabel('m (emu/g)')
    plt.title('Magnetización sin señal diamagnética')
    plt.legend()
    plt.show()

#%%
    param_deriv=fit.print_pars(ret=True)
    #Se guarda Ms y <mu>
    sus_diamag.append(fit.params['C'])
    mag_sat.append(param_deriv[-1]/concentracion_MNP[k]*1000)
    mean_mu_mu.append(param_deriv[2])
# %%

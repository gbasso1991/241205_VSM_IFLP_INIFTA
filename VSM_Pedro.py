#%%!/usr/bin/env python3

"""
@author: pedro
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/media/pedro/Datos/py')
from mlognormfit import fit3
from mvshtools import mvshtools as mt
from uncertainties import ufloat


#%%Se abre del archivo con el ciclo. El VSM Lakeshore tiene 12 lineas de encabezado
os.chdir('/home/giuliano/Documentos/Doctorado/Datos_ESAR_24/241205_VSM_IFLP_INIFTA')
# nombre_archivo       =['1b.txt','2b1.txt','4b2.txt','5b.txt','7b1.txt','9b1.txt','10b.txt','11b1.txt','13b.txt','15b.txt']
# masasachet   = np.array([564, 677,528,608,618,561,672,680,594,633,580])10*-4  #g
# masasachetNP = np.array([900,861,868,809,925,850])10*-4  #g
# masaFF=masasachetNP-masasachet
# concentracion_MNP    =[0.082, 0.4, 0.20, 0.36, 0.28, 0.22, 0.054, 0.17, 0.14, 0.030] #mg/ml
# err_conentracion_MNP =[0.009, 0.1, 0.02, 0.05, 0.03, 0.04, 0.008, 0.06, 0.03, 0.005] #mg/ml
colores = ['blue', 'green', 'red', 'black', 'magenta', 'orange', 'cyan', 'brown', 'purple', 'lime']
#%%
# Definir la clase Muestra
class Muestra:
    def _init_(self, nombre_archivo, masa_sachet, masa_sachetNP, concentracion_MNP, err_concentracion_MNP):
        self.nombre_archivo = nombre_archivo
        self.masa_sachet = masa_sachet
        self.masa_sachetNP = masa_sachetNP
        self.concentracion_MNP = concentracion_MNP
        self.err_concentracion_MNP = err_concentracion_MNP

#%% Crear una lista vacía para almacenar las muestras
muestras = []

# Agregar muestras manualmente (sin vectores previos)
muestras.append(Muestra('LB100_CP2.txt',  469, 961, 1, 0.01)) 
muestras.append(Muestra('LB100_FP.txt', 603, 1093, 0.85  , 0.01  )) 


# Agrega las demás muestras siguiendo el mismo formato
# muestras.append(Muestra(...))

# Crear vectores a partir de los objetos en la lista
nombre_archivo  = [muestra.nombre_archivo                 for muestra in muestras]
tubo            = np.array([muestra.tubo                  for muestra in muestras])
masasachet      = np.array([muestra.masa_sachet           for muestra in muestras])*1e-4
masasachetNP    = np.array([muestra.masa_sachetNP         for muestra in muestras])*1e-4
concentracion_MNP = np.array([muestra.concentracion_MNP   for muestra in muestras])
errores         = np.array([muestra.err_concentracion_MNP for muestra in muestras])
masaFF          = masasachetNP-masasachet


#Se definen variables que se usarán en los ajustes

contribución_lineal=np.zeros(len(nombre_archivo))

sus_diamag=[]
offset=[]
ms_lin=[]
magsat=[]
magsat2=[]
meanmu_mu=[]
meanmu_mu2=[]
mu0_fit2=[]
sig0_fit2=[]

#Sesión con curva anhisteretica
qq = []
#Sesion con curva anhisterética sin señal diamagnética
qq2 = []

#Iniciamnos sesión de ajuste (aun no se ajusta) y graficamos curvas originales normalizadas

for k in range(len(nombre_archivo)):
    #Se lee el archivo
    archivo=nombre_archivo[k]
    data = np.loadtxt (archivo, skiprows=12);
    campo = data[:,0]; momento = data[:,1]
    #Normalizamos por masa de FF
    magnetizacion_FF=momento/masaFF[k] # /concentracion_MNP[k]*1000 #(emu/g)
           
    #Armamos la curva anhisteretica
    a = mt.anhysteretic(campo,magnetizacion_FF)

    campo_anhist=a[0]
    magnetizacion_FF_anhist=a[1]
    
    # #Se inicia la sesión de ajuste con la curva anhisterética
    qq.append(fit3.session (campo_anhist,magnetizacion_FF_anhist, fname='anhi'))
    q = qq[k]
    q.label=archivo

    #Gráficamos el ciclo antes de ajustar
    if k==0:
        plt.figure(111)
        plt.clf()
        plt.figure(112)
        plt.clf()
        plt.figure(113)
        plt.clf()
        plt.figure(114)
        plt.clf()
       
    plt.figure(111)
    plt.plot(campo, magnetizacion_FF,'ro',label=q.label,color=colores[k])
    plt.xlabel('Campo (Oe)')
    plt.ylabel('Magnetizacion (emu/g)')
    plt.title('Normalizado por masa de FF')
    plt.legend()
    plt.savefig('Ciclos_normalizados_por_masa_de_FF')
    plt.show()

# Se realizan los ajustes  

    # q = qq[k]
    # archivo=nombre_archivo[k]
    # q.label=archivo
    # #Se inicia la sesión de ajuste
    
    q.fix('sig0')
    q.fix('mu0')
    q.free('dc')
    q.fit()
    q.update()
    #q.free('sig0')
    #q.free('mu0')
    q.set_yE_as('sep')
    q.fit()
    q.update()
    #q.save()
    q.print_pars()
    # q.fix('C',)
    # q.setp('C',-9e-7)
    # #q.set_yE_as('sep')
    # q.fit()
    # q.update()
    # q.save()
    
    plt.figure(112)
    plt.plot(campo_anhist, (magnetizacion_FF_anhist-q.params['C'].value*campo_anhist-q.params['dc'])/concentracion_MNP[k]*1000,'ro',label=q.label,color=colores[k])
    plt.xlabel('Campo (Oe)')
    plt.ylabel('Momento (emu/g)')
    plt.title('Magnetización sin señal diamagnétcia normalizado por massa de MNP')
    plt.legend()
    plt.show()
    
    plt.figure(114)
    plt.plot(campo_anhist, (magnetizacion_FF_anhist-q.params['C'].value*campo_anhist-q.params['dc']),'ro',label=q.label,color=colores[k])
    plt.xlabel('Campo (Oe)')
    plt.ylabel('Momento (emu/g)')
    plt.title('Magnetización sin señal diamagnétcia')
    plt.legend()
    plt.show()
    
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    derparam=q.print_pars(par='True')
    #Se guarda Ms y <mu>
    sus_diamag.append(q.params['C'])
    magsat.append(derparam[-1]/concentracion_MNP[k]*1000)
    meanmu_mu.append(derparam[2])
    #offset.append(ufloat(b['offset'].value, b['offset'].stderr))
    #ms_lin.append(ufloat(b['Ms'].value, b['Ms'].stderr))
    
    y_norm=(magnetizacion_FF_anhist-q.params['C'].value*campo_anhist-q.params['dc'])/magsat[k].nominal_value/concentracion_MNP[k]
   
    plt.figure(113)
    plt.plot(campo_anhist, y_norm,'ro',label=q.label,color=colores[k])
    plt.xlabel('Campo (Oe)')
    plt.ylabel('Momento (emu/g)')
    plt.title('Magnetización sin señal diamagnétcia normalizado por Ms de MNP y por concentracion')
    plt.legend()
    plt.show()

   
#   #Analizamos las medidas sin señal diamagnética y normalizadas por concetracion
    y_new= (magnetizacion_FF_anhist-q.params['C'].value*campo_anhist-q.params['dc'])/concentracion_MNP[k]*1000


    qq2.append(fit3.session (campo_anhist,y_new, fname='anhi_normMs'))
    q2 = qq2[k]
    archivo=nombre_archivo[k]
    q2.label=archivo

    q2.fix('C',)
    q2.setp('C',0)
    q2.fix('sig0')
    q2.fix('mu0')
    q2.free('dc')
    q2.fit()
    q2.update()
    q2.free('sig0')
    q2.free('mu0')
    q2.fit()
    q2.update()
    q2.print_pars()
    q2.save()
    
 
    derparam2=q2.print_pars(par='True')
    meanmu_mu2.append(derparam2[2])
    magsat2.append(derparam2[-1])

    mu0_fit2.append(ufloat(qq2[k].params['mu0'].value,qq2[k].params['mu0'].stderr))
    sig0_fit2.append(ufloat(qq2[k].params['sig0'].value,qq2[k].params['sig0'].stderr))


plt.figure(111)
plt.savefig('Ciclos_normalizados_por_masa_de_FF')

plt.figure(112)
plt.savefig('Ciclo_sin_señal_diamagnetcia_normalizado_por_massa_de_MNP')
 
plt.figure(114)
plt.savefig('Ciclo_sin_señal_diamagnetica')
 
plt.figure(113)
plt.savefig('Ciclo_sin_señal_diamagnetcia_normalizado_por_Ms_de_MNP_y_por_concentracion')




# Extraer el valor nominal y la desviación estándar
nominal_values_susdiamag = [v.value for v in sus_diamag]
std_devs_susdamag        = [v.stderr for v in sus_diamag]

# Extraer el valor nominal y la desviación estándar
nominal_values_meanmu = [v.nominal_value for v in meanmu_mu]
std_devs_meanmu       = [v.std_dev for v in meanmu_mu]

# Extraer el valor nominal y la desviación estándar
nominal_values_Ms = [v.nominal_value for v in magsat]
std_devs_Ms       = [v.std_dev for v in magsat]


# Crear un gráfico
plt.figure(121)
plt.clf()
plt.title('SUSCEPTIBILIDAD')
plt.errorbar(tubo,nominal_values_susdiamag, yerr=std_devs_susdamag, fmt='o')
plt.xlabel('Numero de tubo')
plt.ylabel('susceptibilidad diamagnética')
plt.legend()
plt.savefig('susceptibilidad')
plt.show()

# Crear un gráfico MS
plt.figure(124)
plt.clf()
plt.title('MAGNETIZACIÓN DE SATURACIÓN')
plt.errorbar( tubo, nominal_values_Ms, yerr=std_devs_Ms, fmt='ro')
plt.xlabel('Numero de tubo')
plt.ylabel('Ms (emu/g)')
plt.savefig('Magnetizacion_de_saturacion')
plt.legend()


#Se necesita ajuste numero 2

# Extraer el valor nominal y la desviación estándar
nominal_values_Ms2 = [v.nominal_value for v in magsat2]
std_devs_Ms2       = [v.std_dev for v in magsat2]

# Extraer el valor nominal y la desviación estándar
nominal_values_meanmu2 = [v.nominal_value for v in meanmu_mu2]
std_devs_meanmu2       = [v.std_dev for v in meanmu_mu2]

# Extraer el valor nominal y la desviación estándar
nominal_values_mu02 = [v.nominal_value for v in mu0_fit2]
std_devs_mu02       = [v.std_dev for v in mu0_fit2]

# Extraer el valor nominal y la desviación estándar
nominal_values_sig02 = [v.nominal_value for v in sig0_fit2]
std_devs_sig02       = [v.std_dev for v in sig0_fit2]


# Crear un gráfico MEAN MU
plt.figure(118)
plt.clf()
plt.title('VALOR MEDIO DEL MOMENTO')
plt.errorbar( tubo, nominal_values_meanmu2, yerr=std_devs_meanmu2, fmt='o')
plt.xlabel('Numero de tubo')
plt.ylabel('<mu>_mu2')
plt.legend()
plt.savefig('Momento_medio_pesado_por_mu')
plt.show()


# Crear un gráfico MU0
plt.figure(119)
plt.clf()
plt.title('Mu 0')
plt.errorbar(tubo, nominal_values_mu02, yerr=std_devs_mu02, fmt='o')
plt.xlabel('Numero de tubo')
plt.ylabel('mu_0')
plt.legend()
plt.savefig('Mu0')
plt.show()


# Crear un gráfico SIG0
plt.figure(122)
plt.clf()
plt.title('SIGMA 0')
plt.errorbar(tubo, nominal_values_sig02, yerr=std_devs_sig02, fmt='o')
plt.xlabel('Numero de tubo')
plt.ylabel('sig_0')
plt.legend()
plt.savefig('Sig0')
plt.show()


# Crear un gráfico MS
plt.figure(124)
plt.errorbar( tubo, nominal_values_Ms2, yerr=std_devs_Ms2, fmt='o')
plt.xlabel('Numero de tubo')
plt.ylabel('Ms (emu/g)')
plt.savefig('Magnetizacion_de_saturacion')
plt.legend()
# %%

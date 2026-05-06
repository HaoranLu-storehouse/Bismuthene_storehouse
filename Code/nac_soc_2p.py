from ase.io import read
from gpaw import GPAW, PW,FermiDirac,Mixer,restart
from gpaw.directmin.etdm import ETDM
from gpaw.directmin.tools import excite
from gpaw.eigensolvers import CG
import numpy as np
from gpaw.utilities.ps2ae import PS2AE
from ase.units import Bohr,Ry
import sys
import os
import multiprocessing as mp
import time
import cmath
import math
 
def nac_cal_kernel(k1,k2,i,j,phi_old_i,phi_old_j,phi_new_i,phi_new_j):
    start=time.time()

    DD_matrix0=np.zeros((2,4),dtype=complex)
    DD_matrix1=np.zeros((2,4),dtype=complex)

    time1=time.time()

    DD_forward1=np.sum(phi_old_i[:,:,:,:].conjugate()*phi_new_j[:,:,:,:])
    DD_matrix0[0,0]=DD_forward1
        
    DD_backward1=np.sum(phi_new_i[:,:,:,:].conjugate()*phi_old_j[:,:,:,:])
    DD_matrix0[0,1]=DD_backward1


    DD_old1=np.sum(phi_old_i[:,:,:,:].conjugate()*phi_old_j[:,:,:,:])
    DD_matrix0[0,2]=DD_old1

    DD_new1=np.sum(phi_new_i[:,:,:,:].conjugate()*phi_new_j[:,:,:,:])
    DD_matrix0[0,3]=DD_new1


    time1=time.time()
    #print(time1-start)

    if k1==k2 and j==i:
            DD=complex(0,0)
        #else:
            #DD=(DD_matrix0[0,0]-DD_matrix0[0,1])
        #DD1=(abs(abs(DD_matrix0[0,0])**2-abs(DD_matrix0[0,1])**2)**0.5)
        #DD1=abs(DD1)**0.5
    else:       

        DD_matrix0[0,0]=complex(abs(DD_matrix0[0,0].real),abs(DD_matrix0[0,0].imag))
        DD_matrix0[0,1]=complex(abs(DD_matrix0[0,1].real),abs(DD_matrix0[0,1].imag))
        DD_matrix0[0,2]=complex(abs(DD_matrix0[0,2].real),abs(DD_matrix0[0,2].imag))
        DD_matrix0[0,3]=complex(abs(DD_matrix0[0,3].real),abs(DD_matrix0[0,3].imag))

        if abs(DD_matrix0[0,0].real) < abs(DD_matrix0[0,0].imag):
            DD_matrix0[0,0]=complex(abs(DD_matrix0[0,0].imag),abs(DD_matrix0[0,0].real))
        if abs(DD_matrix0[0,1].real) < abs(DD_matrix0[0,1].imag):
            DD_matrix0[0,1]=complex(abs(DD_matrix0[0,1].imag),abs(DD_matrix0[0,1].real))
        if abs(DD_matrix0[0,2].real) < abs(DD_matrix0[0,2].imag):
            DD_matrix0[0,2]=complex(abs(DD_matrix0[0,2].imag),abs(DD_matrix0[0,2].real))
        if abs(DD_matrix0[0,3].real) < abs(DD_matrix0[0,3].imag):
            DD_matrix0[0,3]=complex(abs(DD_matrix0[0,3].imag),abs(DD_matrix0[0,3].real))    



        DD1=DD_matrix0[0,0]-DD_matrix0[0,2]
        DD2=DD_matrix0[0,3]-DD_matrix0[0,1]

        DD=DD1*np.sign(DD1.real)-DD2*np.sign(DD2.real)

    
        time1=time.time()
    print(k1,k2,i,j,time1-start)

    return DD





def cal_nac(k1,k2,i,j,sufx):
    #start=time.time()
    if k1 <= k2 :
        phi_old_i=wave_old[i-bmin,k1-1,:,:,:,:]

        time1=time.time()
        #print(time1-start)

        phi_old_j=wave_old[j-bmin,k2-1,:,:,:,:]

        phi_new_i=wave_new[i-bmin,k1-1,:,:,:,:]          

        phi_new_j=wave_new[j-bmin,k2-1,:,:,:,:]             

        DD = nac_cal_kernel(k1,k2,i,j,phi_old_i,phi_old_j,phi_new_i,phi_new_j)  

        mag = np.zeros((1,4),dtype=complex)
        mag[0,0]=np.sum(phi_old_j[0,:,:,:]*phi_old_j[0,:,:,:].conjugate())
        mag[0,1]=np.sum(phi_old_j[0,:,:,:]*phi_old_j[1,:,:,:].conjugate())
        mag[0,2]=np.sum(phi_old_j[1,:,:,:]*phi_old_j[0,:,:,:].conjugate())
        mag[0,3]=np.sum(phi_old_j[1,:,:,:]*phi_old_j[1,:,:,:].conjugate())

        fl=open('mag_%01d_%01d' %(k2,j),'w+')
        print(k2,j,end='   ',file=fl)
        for p in range(4):
            print(mag[0,p].real,end='   ',file=fl)
            print(mag[0,p].imag,end='   ',file=fl)
        print('',file=fl)
        fl.close()

    
    else:
        DD = 0
    

    V_ik1_jk2=-0.658218/2/step*DD 

    return V_ik1_jk2




if __name__ == '__main__':
    start_main=time.time()
    bmin=int(sys.argv[1])
    bmax=int(sys.argv[2])
    nkpt=int(sys.argv[3])
    step=int(sys.argv[4])
    sufx=int(sys.argv[5])
    nband=bmax-bmin+1
    band_range=range(bmin,bmax+1)
    loops=np.array([])
    
    calc_old = GPAW('../wave_old.gpw')
    calc_new = GPAW('../wave_new.gpw')
    u=calc_old.wfs.get_wave_function_array(0,0, 0,cut=False, realspace=True, periodic=True)
    wave_old=np.zeros((nband,nkpt,u.shape[0],u.shape[1],u.shape[2],u.shape[3]),dtype=complex)
    wave_new=np.zeros((nband,nkpt,u.shape[0],u.shape[1],u.shape[2],u.shape[3]),dtype=complex)
    phase_old=np.zeros((nkpt,u.shape[0],u.shape[1],u.shape[2],u.shape[3]),dtype=complex)
    phase_new=np.zeros((nkpt,u.shape[0],u.shape[1],u.shape[2],u.shape[3]),dtype=complex)
    phase1=np.zeros((nkpt,u.shape[0],u.shape[1],u.shape[2],u.shape[3]),dtype=complex)


    energy = np.zeros((1,0))
    energy_band=np.zeros((nkpt,bmax+1))
    for k in range(nkpt):
        eig=calc_new.get_eigenvalues(k,0)
        eig=eig[bmin-1:bmax].reshape(1,-1)
        #fl=open('energy_k%01d' %k,'a')
        #np.savetxt(fl,eig)
        #fl.close()
        energy=np.hstack((energy,eig))
        for j in band_range:
            energy_band[k,j]=eig[0,j-bmin]

    fl=open('energy','a')
    np.savetxt(fl,energy)
    fl.close()


    for k1 in range(nkpt):
        phase_old[k1,:,:,:,:]=calc_old.wfs.get_wave_function_array(0, k1, 0,cut=False, realspace=True, periodic=True)
        phase_new[k1,:,:,:,:]=calc_new.wfs.get_wave_function_array(0, k1, 0,cut=False, realspace=True, periodic=True)
        #phase1[k1,:,:,:,:] = phase_old[k1,:,:,:,:].conjugate()/np.linalg.norm(phase_old[k1,:,:,:,:])*phase_new[k1,:,:,:,:].conjugate()/np.linalg.norm(phase_new[k1,:,:,:,:])
        
    
    for i in band_range:
        for k1 in range(nkpt):    
            wave_old[i-bmin,k1,:,:,:,:]=calc_old.wfs.get_wave_function_array(i-1, k1, 0,cut=False, realspace=True, periodic=False)
            wave_new[i-bmin,k1,:,:,:,:]=calc_new.wfs.get_wave_function_array(i-1, k1, 0,cut=False, realspace=True, periodic=False)
        #wave_old[i-bmin,:,:,:,:,:]=wave_old[i-bmin,:,:,:,:,:]*phase1[:,:,:,:,:]
        #wave_new[i-bmin,:,:,:,:,:]=wave_new[i-bmin,:,:,:,:,:]*phase1[:,:,:,:,:]
        for k1 in range(nkpt):
            wave_old[i-bmin,k1,:,:,:,:]=wave_old[i-bmin,k1,:,:,:,:]/np.linalg.norm(wave_old[i-bmin,k1,:,:,:,:])
            wave_new[i-bmin,k1,:,:,:,:]=wave_new[i-bmin,k1,:,:,:,:]/np.linalg.norm(wave_new[i-bmin,k1,:,:,:,:])
        
    for k1 in range(nkpt): 
        for i in band_range: 
            for k2 in range(nkpt):
                for j in band_range:
                    a=np.array([k1+1,k2+1,i,j,sufx])
                    loops=np.append(loops,a)
    calc_old.close()
    calc_new.close()

#    
    
    loops=loops.reshape(-1,5)
    #print(loops)
    #for row in loops:
    #    print(row[0],row[1],row[2],row[3],row[4])


    time1=time.time()
    #print(time1-start_main)

    pool=mp.Pool(21)
    #loops=np.array([(k1+1,k2+1,i+1,j+1,sufx) for k1,k2,i,j in nkpt,nkpt,band_range,band_range] )
    results = pool.starmap(cal_nac, [(int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4])) for row in loops])
    
    pool.close()
    os.system("cat mag* >> MagP%04d" %sufx)
    os.system("rm -rf mag* ")
    results=np.array(results).reshape(nband*nkpt,-1)
    #print(results.shape)

    for i in range(nband*nkpt):
        for j in range(nband*nkpt):
            if j < i :
                results[i,j]=-results[j,i]
    real=results.real.reshape(-1,1)
    imag=results.imag.reshape(-1,1)
    NAC=np.hstack((real,imag)).reshape(-1,nband*nkpt*2)
    np.savetxt('PSnac%04d' %sufx,NAC,fmt='%14.5E')
    end_main=time.time()
    print(end_main-start_main)




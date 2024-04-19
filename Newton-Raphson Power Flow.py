# Developed by HyunSu Shin Apr. 2022
# This Code is developed and tested on Python 3.8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data 불러오기
Bus = np.genfromtxt('33Bus.txt') 
Line = np.genfromtxt('33Line.txt')

# Bus variable
Bus_num, Bus_type = Bus[: , 0], Bus[: , 1]
PG, QG, PL, QL, VM, Angle = Bus[: , 2], Bus[: , 3], Bus[: , 4], Bus[: , 5], Bus[: , 6], Bus[: , 7]
P, Q = np.subtract(PG, PL), np.subtract(QG, QL)

# Line Variable
From, To = np.array(Line[0:len(Line),0],dtype=np.int64,)-1, np.array(Line[0:len(Line),1],dtype=np.int64)-1
LR, LX, LB = Line[: , 2], 1j*Line[: , 3], 1j/2*Line[: , 4]
Z = np.add(LR, LX) #32개
IZ = np.divide(1, Z)
PV_num = sum(Bus_type==2)
PQ_num = sum(Bus_type==3)
 
# Ybus
Ybus = np.zeros((len(Bus_num),len(Bus_num)), dtype=complex) 
for i in range(len(Line)):
    Ybus[From[i], From[i]] += IZ[i] + LB[i]
    Ybus[To[i], To[i]] += IZ[i] + LB[i]
    Ybus[From[i], To[i]] = -1 * IZ[i]
    Ybus[To[i], From[i]] = -1 * IZ[i]

G, B = np.real(Ybus), np.imag(Ybus)

## Ybus_M, A for Polar
# Ybus_M , Ybus_A = np.zeros((len(Bus_num),len(Bus_num))), np.zeros((len(Bus_num),len(Bus_num)))
# for i in range(len(Bus_num)):
#     for j in range(len(Bus_num)):
#         Ybus_M[i,j] = np.abs(Ybus[i,j])
#         Ybus_A[i,j] = np.angle(Ybus[i,j])
#%%
# iteration
iteration = 0
while True:    
    # P, Q equation
    Pcal, Qcal = np.zeros(len(Bus_num)), np.zeros(len(Bus_num))
    for i in range(len(Bus_num)):
        for j in range(len(Bus_num)):
            Pcal[i] += VM[i] * VM[j] * (G[i,j] * np.cos(Angle[i]-Angle[j]) + B[i,j] * np.sin(Angle[i]-Angle[j]))        #Rectangular
            Qcal[i] += VM[i] * VM[j] * (G[i,j] * np.sin(Angle[i]-Angle[j]) - B[i,j] * np.cos(Angle[i]-Angle[j]))
            # Pcal[i] += VM[i] * Ybus_M[i,j] * VM[j] * np.cos(Angle[i]-Angle[j]-Ybus_A[i,j])                            #Polar
            # Qcal[i] += VM[i] * Ybus_M[i,j] * VM[j] * np.sin(Angle[i]-Angle[j]-Ybus_A[i,j])                         
    
# calculate ΔP, ΔQ
    delP = np.subtract(P[1:len(Bus_num)], Pcal[1:len(Bus_num)])                                                         # PV+PQ버스 개수
    delQ = np.subtract(Q[1+PV_num:len(Bus_num)], Qcal[1+PV_num:len(Bus_num)])                                           # PV버스 개수
        
# Jacobian matrix (Rectangular)
    J1 = np.zeros((PQ_num+PV_num, PQ_num+PV_num))
    J2 = np.zeros((PQ_num+PV_num, PQ_num))
    J3 = np.zeros((PQ_num, PQ_num+PV_num))
    J4 = np.zeros((PQ_num, PQ_num))

# J1, J2
    for i in range(1, len(Bus_num)):                                                                                    # 행 개수 일치
        for j in range(1, len(Bus_num)):
            if i == j:
                J1[i-1,j-1] = -1.0 * Qcal[i] -1.0 * B[i,j] * np.power(VM[i],2)
            else:
                J1[i-1,j-1] = VM[i] * VM[j] * (G[i,j] * np.sin(Angle[i]-Angle[j]) - B[i,j] * np.cos(Angle[i]-Angle[j])) 
        for j in range(1+PV_num, len(Bus_num)):
            if i == j:
                J2[i-1,j-1-PV_num] = Pcal[i] / VM[i] + G[i,j] * VM[j]
            else:
                J2[i-1,j-1-PV_num] = VM[i] * (G[i,j] * np.cos(Angle[i]-Angle[j]) + B[i,j] * np.sin(Angle[i]-Angle[j]))

# J3, J4
    for i in range(1+PV_num, len(Bus_num)):
        for j in range(1, len(Bus_num)):
            if i == j:
                J3[i-1-PV_num,j-1] = Pcal[i] -1.0 * G[i,j] * np.power(VM[i],2)
            else:
                J3[i-1-PV_num,j-1] = -1.0 * VM[i] * VM[j] * (G[i,j] * np.cos(Angle[i]-Angle[j]) + B[i,j] * np.sin(Angle[i]-Angle[j]))
        for j in range(1+PV_num, len(Bus_num)):
            if i == j:
                J4[i-1-PV_num,j-1-PV_num] = Qcal[i] / VM[i] - B[i,j] * VM[i]
            else:
                J4[i-1-PV_num,j-1-PV_num] = VM[i] * (G[i,j] * np.sin(Angle[i]-Angle[j]) - B[i,j] * np.cos(Angle[i]-Angle[j]))
        
    J = np.vstack([np.hstack([J1,J2]), np.hstack([J3,J4])])
    
# for iterative method
    delPQ = np.transpose(np.hstack((delP, delQ)))
    InvJ = np.linalg.inv(J)
    delx = np.dot(InvJ,delPQ)
    delA = delx[:PQ_num+PV_num] #PV, PQ 모선 개수
    # delA = np.hstack([Angle[:len(Bus_num)-PQ_num-PV_num], delA])
    delV = delx[PQ_num+PV_num:] #PQ 모선 개수
    # delV = np.hstack([np.zeros(len(Bus_num)-PQ_num), delV])
    
    for i in range(1,len(Bus_num)):
        Angle[i] += delA[i-1]
        if Bus_type[i] == 3:
            VM[i] += delV[i-1-PV_num]
        
# Convergence judgment
    error = np.max(np.abs(delPQ))
    iteration += 1
    
    if error < 1e-6:
        break
 
# Result    
PG[0] = Pcal[0] + PL[0]
for i in range(0, len(Bus_num)):
    if Bus_type[i] != 3:
        QG[i] = Qcal[i] + QL[i]   
    
Angle = np.rad2deg(Angle) 

Bus[:,0] = Bus[:,0].astype(int)
Bus[:,1] = Bus[:,1].astype(int)
Bus[:,2] = np.round(PG,6)
Bus[:,3] = np.round(QG,6)
Bus[:,4] = np.round(PL,6)
Bus[:,5] = np.round(QL,6)
Bus[:,6] = np.round(VM,6)
Bus[:,7] = np.round(Angle,6)

# Ploss
Ploss = np.round(PG[0]+sum(PG[Bus_type==2])-sum(PL[Bus_type==3]),5)

print(f"iteration : {iteration}")
print(f"Ploss : {Ploss}")

df1 = pd.DataFrame(Bus, columns=['Bus_num', 'Bus_type', 'PG', 'QG', 'PL', 'QL', 'V', 'theta'])
df1.to_csv('NRPF_result.csv', encoding='utf-8', index=False)









# -------------- Jacobian matrix (Polar) ----------------
#     for i in range(PQ_num+PV_num):
#         for j in range(PQ_num+PV_num):
#             if i == j:
#                 J1[i,j] = -1 * Qcal[i] - VM[i] * Ybus_M[i,j] * VM[j] * np.sin(Ybus_A[i,j])            
#             else:
#                 J1[i,j] = VM[i] * Ybus_M[i,j] * VM[j] * np.sin(Angle[i]-Angle[j]-Ybus_A[i,j])
#         for j in range(PQ_num):
#             if i == j:                
#                 J2[i,j] = VM[i] * Ybus_M[i,i] * np.cos(Ybus_A[i,i]) + Pcal[i] / VM[i]
#             else:
#                 J2[i,j] = VM[i] * Ybus_M[i,j] * np.cos(Angle[i]-Angle[j]-Ybus_A[i,j])
                
#     for i in range(PQ_num):
#         for j in range(PQ_num+PV_num):
#             if i == j:
#                 J3[i,j] = Pcal[i] - VM[i] * Ybus_M[i,j] * VM[j] * np.cos(Ybus_A[i,j])
#             else:
#                 J3[i,j] = -1 * VM[i] * Ybus_M[i,j] * VM[j] * np.cos(Angle[i]-Angle[j]-Ybus_A[i,j])
#         for j in range(PQ_num):
#             if i == j:
#                 J4[i,j] = -1 * VM[i] * Ybus_M[i,i] * np.sin(Ybus_A[i,i]) + Qcal[i] / VM[i]
#             else:
#                 J4[i,j] = VM[i] * Ybus_M[i,j] * np.sin(Angle[i]-Angle[j]-Ybus_A[i,j])     
# -----------------------------------------------------------------------------------------------

# --------------------- Ploss --------------------------------
# Ploss = np.zeros(len(Bus_num)-1, dtype=complex)

# for i in range(len(Bus_num)-1):
#     Ploss[i] = VM[From[i]]*(np.cos(Angle[From[i]])+1j*np.sin(Angle[From[i]]))-VM[To[i]]*(np.cos(Angle[To[i]])+1j*np.sin(Angle[To[i]]))
    
    
#     for b in range(len(Bus_num)-1):
#         k[a,b] = VM[a]*np.cos(Angle[a])-VM[b]*np.cos(Angle[b])+1j*(VM[a]*np.sin(Angle[a])-VM[b]*np.sin(Angle[b]))
    

# k = np.zeros((len(Bus_num)-1,len(Bus_num)-1), dtype=complex)
    
# for a in range(len(Bus_num)-1):
#     for b in range(len(Bus_num)-1):
#         k[a,b] = VM[a]*np.cos(Angle[a])-VM[b]*np.cos(Angle[b])+1j*(VM[a]*np.sin(Angle[a])-VM[b]*np.sin(Angle[b]))
#         # if a==b 
# q = (k**2)/(LR)
# --------------------------------------------------------------------------------------------
    

# ---------------------- to.csv -----------------------
# df1 = pd.DataFrame(Bus, columns=['Bus_num', 'Bus_type', 'PG', 'QG', 'PL', 'QL', 'V', 'theta'])
# df2 = pd.DataFrame(iteration, columns=['iteration'])
# df3 = pd.merge(df1,df2,right_index=True)
# df3 = pd.concat([df1,df2], ignore_index=True)
# df1.to_csv('NRPF_result.csv', encoding='utf-8', index=False)
# ----------------------------------------


# ---------------------- plot ----------------------------------
plt.figure(figsize=(20,10))
plt.title('NR-PF Result')
fig, ax1 = plt.subplot()
ax1.plt.plot(Bus_num,VM,'b', label='Vpu')
ax2 = ax1.twinx()
ax2.plt.plot(Bus_num,Angle,'r', label='Theta')
plt.legend(loc='best')
plt.show
# ----------------------------------------
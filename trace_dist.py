#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:12:41 2019

@author: sudhang
"""
import sys
import numpy as np
from numpy import linalg as la
import math 
import cmath
import matplotlib.pyplot as plt
print("Enter dimension of Hamiltonian")
D = int(input())  
ham = [] 

print("Enter the entries in a single line (separated by space): ")     
ham = np.array(list(map(int, input().split()))).reshape(D,D) 
E_val , E_vec = la.eig(ham)
print("The Hamiltionian is:")
print(ham)
print("Its eigen values are:")
print(E_val)
print("Corresponding eigen vectors are:")
print(E_vec)
print("Enter the coefficients of pure state, separated by spaces")
coef=np.array(list(map(float,input().split()))).reshape(D,1)
print("The state is:")
print(coef)
abcd0=np.zeros((D,1)).reshape(D,1)
abcd0=la.solve(E_vec,coef)
abcd0=abcd0.reshape(D,1)
print("Density matrix for state at t=0 is:")
den0= np.outer(coef , coef)
print(den0)
print("Set time till which state is to be evolved:-")
time=int(input())
print("Enter number of steps on computation:")
steps=int( input())
products=np.zeros((steps,1)).reshape(steps,1)
trace_d=np.zeros((steps,1)).reshape(steps,1)
state0=np.zeros((D,1)).reshape(D,1)
count=0 
for k in range(D):
    state0=state0+abcd0[k]*(E_vec[:,k].reshape(D,1))
for t in np.linspace(0,time,steps):
    state=np.zeros((D,1),dtype=np.complex).reshape(D,1)
    abcd=np.zeros((D,1),dtype=np.complex).reshape(D,1)
    for z in range(D):
        abcd[z]=abcd0[z]*cmath.exp(-1j*E_val[z]*t/2)    
    for k in range(D):
        state=state+abcd[k]*(E_vec[:,k].reshape(D,1))
    den_t=np.outer(np.conj(state) , state).reshape(D,D)
    if (t == 0):
        print(den_t)
    eigv=np.linalg.eig((den0-den_t))[0].reshape(D,1)
    eigv=np.abs(eigv)
    print(eigv)
    trace_d[count]=np.sum(eigv,axis=0)*1/2
    inner_product=np.dot(np.conj(state).T,state0)
    inner_productx=inner_product*(inner_product.conjugate())
    products[count]=inner_productx
    count=count+1
x=np.array(np.linspace(0,time,steps))
y=products
y1=trace_d
plt.subplot(2, 1, 1)

plt.title("Fidelity and trace between evolved and original state")
plt.plot(x,y,'r--')
plt.xlabel('time')
plt.ylabel('Fidelity')
plt.subplot(2, 1, 2)
#plt.ylim()
#plt.title("Trace distance between evolved and original state")
plt.xlabel('time')
plt.ylabel('Trace Distance')
plt.plot(x,y1)
plt.show()
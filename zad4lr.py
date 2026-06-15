# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:27:19 2022

@author: PRz
"""
import hickle as hkl

import numpy as np
import nnet as net
import matplotlib.pyplot as plt
from matplotlib import cm

x1_range = np.linspace(-1, 1, 10)
x2_range = np.linspace(-1, 1, 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y_t_mesh = np.tan((np.pi / 3.0) * X1 * X2)

x = np.vstack([X1.ravel(), X2.ravel()])
y_t = Y_t_mesh.ravel().reshape(1, -1)

max_epoch = 50000
err_goal = 0.01
disp_freq = 2000 
lr = 0.01
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04
mc = 0.9
L = x.shape[0] 
K1 = 10
K2 = 15
K3 = y_t.shape[0] 

SSE_vec = [] 

SSE = 0
lr_vec = list()

w1, b1 = net.nwtan(K1, L)
w2, b2 = net.nwtan(K2, K1)
w3, b3 = net.rands(K3, K2)

fig = plt.figure(figsize=(18, 5))

for epoch in range(1, max_epoch+1): 
    y1 = net.tansig( np.dot(w1, x),  b1)
    y2 = net.tansig( np.dot(w2, y1),  b2)
    y3 = net.purelin(np.dot(w3, y2), b3)
    e = y_t - y3
    
    SSE_t_1 = SSE
    SSE = net.sumsqr(e) 
    if np.isnan(SSE):
        break
    else:
        if SSE > er * SSE_t_1:
            lr *= ksi_dec
        elif SSE < SSE_t_1:
            lr *= ksi_inc
    lr_vec.append(lr)

    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3)
    d1 = net.deltatan(y1, d2, w2) 
    dw1, db1 = net.learnbp(x,  d1, lr) 
    dw2, db2 = net.learnbp(y1, d2, lr)
    dw3, db3 = net.learnbp(y2, d3, lr)
   
    w1 += dw1
    b1 += db1
    w2 += dw2
    b2 += db2
    w3 += dw3
    b3 += db3
    
    

    SSE_vec.append(SSE) 
 
    if SSE < err_goal: 
        break 
    if (epoch % disp_freq) == 0: 
        print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
        Y3_mesh = y3.reshape(X1.shape)
        Error_mesh = np.abs(Y_t_mesh - Y3_mesh)
        
        plt.clf() # Czyszczenie poprzedniej konfiguracji okna
        
        # 1. Oryginalna funkcja
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(X1, X2, Y_t_mesh, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_title(f'1. Oryginalna funkcja (Epoka {epoch})')
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')

        # 2. Odtworzenie sieci
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(X1, X2, Y3_mesh, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax2.set_title(f'2. Odtworzenie sieci | SSE: {SSE:.4f}')
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')

        # 3. Wykres błędu
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_surface(X1, X2, Error_mesh, cmap=cm.plasma, linewidth=0, antialiased=False)
        ax3.set_title('3. Błąd bezwzględny $|y_t - y_3|$')
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')

        plt.tight_layout()
        plt.draw()
        plt.pause(1e-2)

print("\nKoniec nauki. Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
        
fig_sse = plt.figure()
plt.plot(SSE_vec) 
plt.ylabel('SSE') 
plt.yscale('log') 
plt.title('Historia SSE globalnie') 
plt.grid(True) 
plt.show()
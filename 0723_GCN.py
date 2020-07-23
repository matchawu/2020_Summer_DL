# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:23:01 2020

@author: wwj
"""

#%%

# 定義鄰居矩陣A，故意使它不對稱，有向圖比較容易看到訊息傳遞的狀況
import numpy as np

A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 0, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

#%%
# 畫圖程式(optional)
import networkx as nx
import matplotlib.pyplot as plt
G=nx.DiGraph(A)
nx.draw(G, with_labels=True)
plt.show()

# 定義初始特徵
X = np.matrix([[ 100., -100.],
                [ 1., -1.],
                [ 2., -2.],
                [ 3., -3.]])

A * X

#%%

D = np.diag(np.array(np.sum(A, 1)).flatten())
L = D-A

L*X

#%%

# 增加自環
I = np.diag([1]*A.shape[0])
A_hat = A + I

A_hat

#%%

A_hat * X

#%%

D_hat = np.diag(np.array(np.sum(A, 1)).flatten())
L_hat = D_hat-A_hat

L_hat*X

#%%


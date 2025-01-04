import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def construct_D_matrices(M, N):
    MN = M*N 
    #D_x
    D_x = (np.eye(MN)*-1)+ np.eye(MN,k=1)
    for i in range(N-1,MN,N):
        D_x[i] = np.zeros(MN)
    #Dy
    D_y = np.zeros((MN,MN))   
    for i in range(MN-N):
        D_y[i,i] = -1
        D_y[i,i+N] = 1
    return D_x,D_y


def col_stack(x):
    return x.flatten()

M,N = (513,513)
D_x,D_y = construct_D_matrices(M,N)

for idx,X in enumerate(['X1','X2','X3']):
    Xi = loadmat(f'{X}.mat')[X]
    x = col_stack(Xi)
    y_x = (D_x@x).reshape(M,N)
    y_y = (D_y@x).reshape(M,N)
    plt.figure(figsize=(10, 5))

    # Plot y_x
    plt.subplot(1, 2, 1)
    plt.title(f"$y_x$ for X{idx}")
    plt.imshow(y_x, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Plot y_y
    plt.subplot(1, 2, 2)
    plt.title(f"$y_y$ for X{idx}")
    plt.imshow(y_y, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Save the plot
    filename = f"plots/derivatives_X{idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
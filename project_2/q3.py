import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def construct_D_matrices(M, N):
    MN = M*N 
    #D_x
    D_x = (np.eye(MN)*-1)+ np.eye(MN,k=1)
    for i in range(N-1,MN,N):
        D_x[i, i + 1] = 0    #Dy
    D_y = np.zeros((MN,MN))   
    for i in range(MN-N):
        D_y[i,i] = -1
        D_y[i,i+N] = 1
    return D_x,D_y

import numpy as np
from scipy.sparse import eye, diags, lil_matrix

def construct_D_matrices_sparse(M, N):
    MN = M * N
    # Construct sparse D_x
    diagonals_x = [-np.ones(MN), np.ones(MN - 1)]
    offsets_x = [0, 1]
    D_x = diags(diagonals_x, offsets_x, shape=(MN, MN), format='lil')
    
    # Remove connections across row boundaries
    for i in range(N - 1, MN, N):
        D_x[i,i] = 0
        if i+1==MN:
            break
        D_x[i,i+1] = 0        
    D_x = D_x.tocsr()  # Convert to CSR format for efficient operations
    
    # Construct sparse D_y
    D_y = lil_matrix((MN, MN))
    for i in range(MN - N):
        D_y[i, i] = -1
        D_y[i, i + N] = 1
    D_y = D_y.tocsr()  # Convert to CSR format for efficient operations

    return D_x, D_y


def col_stack(x):
    return x.flatten()

M,N = (513,513)
D_x,D_y = construct_D_matrices_sparse(M,N)

for idx,X in enumerate(['X1','X2','X3']):
    Xi = loadmat(f'{X}.mat')[X]
    x = col_stack(Xi)
    y_x = (D_x@x).reshape(M,N)
    y_y = (D_y@x).reshape(M,N)
    Gi = np.sqrt(y_x**2 + y_y**2)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title(f"$X_{idx}$")
    plt.imshow(Xi, cmap='viridis', aspect='auto')
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot y_x
    plt.subplot(1, 4, 2)
    plt.title(f"$\partial_{'x'}X$ for $X_{idx}$")
    plt.imshow(y_x, cmap='viridis', aspect='auto')
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot y_y
    plt.subplot(1, 4, 3)
    plt.title(f"$\partial_{'y'}X$ for $X_{idx}$")
    plt.imshow(y_y, cmap='viridis', aspect='auto')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 4, 4)
    plt.title(f"G for $X_{idx}$")
    plt.imshow(Gi, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")

    # Save the plot
    filename = f"plots/derivatives_X{idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import eye, diags, lil_matrix,hstack,csr_matrix
from scipy import sparse
def construct_D_matrices(M, N):
    MN = M*N 
    #D_x
    D_x = (np.eye(MN)*-1)+ np.eye(MN,k=1)
    for i in range(N-1,MN,N):
        D_x[i,i] = 0
        if i+1==MN:
            break
        D_x[i,i+1] = 0  
    D_y = np.zeros((MN,MN))   
    for i in range(MN-N):
        D_y[i,i] = -1
        D_y[i,i+N] = 1
    return D_x,D_y



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


def construct_D_matrices_3D_sparse(M, N, P):
    MN = M * N  # Number of cells in each 2D slice
    MNP = M * N * P  # Total number of cells in the 3D grid

    # Construct D_x (finite differences along x-axis)
    diagonals_x = [-np.ones(MNP), np.ones(MNP - 1)]
    offsets_x = [0, 1]
    D_x = diags(diagonals_x, offsets_x, shape=(MNP, MNP), format='lil')
    
    for z in range(P):  
        for i in range(N - 1, MN + z * MN, N):
            D_x[i, i] = 0
            if i + 1 < MNP:
                D_x[i, i + 1] = 0
    D_x = D_x.tocsr()  

    D_y = lil_matrix((MNP, MNP))
    for z in range(P):  
        for i in range((M - 1) * N + z * MN):  
            D_y[i, i] = -1
            if i + N < MNP:
                D_y[i, i + N] = 1
    D_y = D_y.tocsr()  

    D_z = lil_matrix((MNP, MNP))
    for i in range(MN * (P - 1)):  
        D_z[i, i] = -1
        D_z[i, i + MN] = 1
    D_z = D_z.tocsr()  

    return D_x, D_y, D_z

def col_stack(x):
    return x.flatten('F')

def create_plots(D_x,D_y,M,N):
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

# M,N = (513,513)
# D_x,D_y = construct_D_matrices_sparse(M,N)
# create_plots(D_x,D_y,M,N)
def Q9():
    A = np.zeros((8,25))#from toy problem

    ray_0 = np.zeros((5,5))
    ray_0[0,:]=1

    # print(ray_0)
    A[0,:] = col_stack(ray_0)

    ray_1 = np.zeros((5,5))
    for i in range(1,5):
        ray_1[i,i-1] = np.sqrt(2)
    # print(ray_1)
    A[1,:] = col_stack(ray_1)

    ray_2 = np.zeros((5,5))
    ray_2[1,:]=1
    # print(ray_2)
    A[2,:] = col_stack(ray_2)

    ray_3 = np.zeros((5,5))
    ray_3[3,:]=1
    # print(ray_3)
    A[3,:] = col_stack(ray_3)

    ray_4 = np.zeros((5,5))
    for i in range(4,-1,-1):
        ray_4[i,4-i] = np.sqrt(2)
    # print(ray_4)
    A[4,:] = col_stack(ray_4)

    ray_5 = np.zeros((5,5))
    ray_5[4,1] = ray_5[3,0] = np.sqrt(2)
    # print(ray_5)
    A[5,:] = col_stack(ray_5)

    ray_6 = np.zeros((5,5))
    ray_6[:,3]=1
    # print(ray_6)
    A[6,:] = col_stack(ray_6)

    ray_7 = np.zeros((5,5))
    for i in range(4,0,-1):
        ray_7[i-1,i] = np.sqrt(2)
    # print(ray_7)
    A[7,:] = col_stack(ray_7)

    M,N = (5,5)
    D_x,D_y = construct_D_matrices(M,N)
    L = np.concatenate((D_x,D_y))
    lambda_reg = 1e-5
    Q = A.T@A +lambda_reg*(L.T@L)
    u,v = np.linalg.eigh(Q)
    kappa = max(u)/min(u)
    # print(f'The condition number of Q is {kappa}\nMax_eig_val:{max(u)}\nMin_eig_val:{min(u)}')
    return A,L

def compute_W_sqrt(L, x, epsilon=1e-5):
    gamma = L @ x  
    W_diag = np.where(np.abs(gamma) >= epsilon, 1.0 / np.abs(gamma), 1.0 / epsilon)
    W = diags(np.sqrt(W_diag))
    return W

def cgls(A, L, y, max_iter=100, eps=1e-6, lambda_reg=1e-5,x_0=[0],irls=False):
    m, n = A.shape
    if x_0[0] != 0:
        x = x_0
    else:
        x = np.zeros(n)  # x_0 is zeros

    sqrt_lambda = np.sqrt(lambda_reg)
    if not irls:
        A = sparse.vstack([A, sqrt_lambda * L])
    else:
        W_sqrt = compute_W_sqrt(L,x)
        A = sparse.vstack([A, (sqrt_lambda)*(W_sqrt@L) ])
        
    if len(y.shape) ==1:
        y = np.expand_dims(y,1)
    y = np.vstack([y, np.zeros((L.shape[0],1))])
    s_k= np.expand_dims(A@x,1) -y
    g_k = A.T @ s_k
    g_0 = g_k.copy()
    d = -g_k  # d_0 = -g_0
    residuals = [np.linalg.norm(g_k)]  # initial gradient norm
    
    for k in range(max_iter):
        #compute Ad_k to simplify for the future
        Ad = A @ d
        #compute step size alpha_k
        alpha = (g_k.T @ g_k) / (d.T@A.T @ Ad)
        #update the solution
        x += (alpha.squeeze() * d).reshape(x.shape)
        s_k += alpha * Ad  # s_{k+1} = s_k + alpha_k A d_k
        g_k_plus_1 = A.T @ s_k

        # check for convergence - gradient norm
        gradient_norm = np.linalg.norm(g_k_plus_1)
        residuals.append(gradient_norm)
        if gradient_norm < eps:
            break
        # compute beta_k for the new search direction and update
        beta = ((g_k_plus_1.T@ g_k_plus_1) / (g_k.T@ g_k)).squeeze()
        d = -g_k_plus_1 + beta * d
        #for next iter
        g_k = g_k_plus_1 
    return x, residuals, k

def Q10_toy():
    A,L = Q9()
    Y = loadmat('Y.mat')['Y']
    y = np.zeros((8))
    # reshaping Y for the numbering of the rays in the pdf
    y[0] = Y[0,4]
    y[1] = Y[1,1]
    y[2] = Y[1,3]
    y[3]=Y[2,2]
    y[4]=Y[3,4]
    y[5]=Y[4,0]
    y[6]=Y[5,1]
    y[7]=Y[6,2]
    x_cgls,res,k = cgls(A,L,y)
    print(k)

def Q11_small():
    import matplotlib 
    matplotlib.use('WebAgg')
    import matplotlib.pyplot as plt
    y = loadmat('Small/y.mat')['y']
    A = loadmat('Small/A.mat')['A']
    M,N,P = (19,19,19)
    D_x,D_y,D_z=construct_D_matrices_3D_sparse(M,N,P)
    L = sparse.vstack([D_x, D_y])
    L = sparse.vstack([L,D_z])
    lambda_reg = 10
    x_cgls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=lambda_reg)
    X_cgls = x_cgls.reshape(M,N,P)
    fig = plt.figure(figsize=(20, 16))

    ax1 = fig.add_subplot(231, projection="3d")
    ax1.set_title(f"Small bag 3D plot, $\lambda$={lambda_reg},Converged in {k} iterations")
    x, y, z = np.where(X_cgls > 0.5)
    scatter=ax1.scatter(x, y, z, c=X_cgls[x, y, z], cmap="viridis", marker="o")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label("Intensity")
    plt.show()

def Q11_large():
    import matplotlib 
    matplotlib.use('WebAgg')
    import matplotlib.pyplot as plt
    y = loadmat('Large/y.mat')['y']
    A = loadmat('Large/A.mat')['A']
    M,N,P = (49,49,49)
    D_x,D_y,D_z=construct_D_matrices_3D_sparse(M,N,P)
    L = sparse.vstack([D_x, D_y])
    L = sparse.vstack([L,D_z])
    lambda_reg=1e-5
    x_cgls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=lambda_reg)
    X_cgls = x_cgls.reshape(M,N,P)
    fig = plt.figure(figsize=(20, 16))

    ax1 = fig.add_subplot(231, projection="3d")
    ax1.set_title(f"Large bag 3D plot, $\lambda$={lambda_reg},Converged in {k} iterations")
    x, y, z = np.where(X_cgls > 0.5)
    scatter=ax1.scatter(x, y, z, c=X_cgls[x, y, z], cmap="viridis", marker="o")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label("Intensity")
    plt.show()

def Q12():
    x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    f1 = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    f2 = np.array([0, 0.0025, 0.0180, 0.1192, 0.5, 0.8808, 0.9820, 0.9975, 1])

    
    _,D_x =construct_D_matrices(9,1) 

    L1_norm_f1 = np.linalg.norm(D_x@f1,1)
    L2_norm_f1 = np.linalg.norm(D_x@f1,2)

    L1_norm_f2 = np.linalg.norm(D_x@f2,1)
    L2_norm_f2 = np.linalg.norm(D_x@f2,2)
    print(f'l1 f1 = {L1_norm_f1}')
    print(f'l2 f1 = {L2_norm_f1}')
    print(f'l1 f2 = {L1_norm_f2}')
    print(f'l2 f2 = {L2_norm_f2}')
    

def Q15():
    import matplotlib 
    matplotlib.use('WebAgg')
    import matplotlib.pyplot as plt
    y = loadmat('Small/y.mat')['y']
    A = loadmat('Small/A.mat')['A']
    M,N,P = (19,19,19)
    D_x,D_y,D_z=construct_D_matrices_3D_sparse(M,N,P)
    L = sparse.vstack([D_x, D_y])
    L = sparse.vstack([L,D_z])
    lambda_reg = 1e-1
    x_cgls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=lambda_reg)

    x_0 = x_cgls
    alpha=0.5/2

    x_irls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=alpha,x_0=x_0,eps=1e-8,irls=True)

    X_irls = x_irls.reshape(M,N,P)
    fig = plt.figure(figsize=(20, 16))

    ax1 = fig.add_subplot(231, projection="3d")
    ax1.set_title(f"Small bag 3D plot of IRLS, alpha={alpha*2},Converged in {k} iterations")
    x, y, z = np.where(X_irls > 0.5)
    scatter=ax1.scatter(x, y, z, c=X_irls[x, y, z], cmap="viridis", marker="o")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label("Intensity")
    plt.show()


def Q16():
    import matplotlib 
    matplotlib.use('WebAgg')
    import matplotlib.pyplot as plt
    y = loadmat('Large/y.mat')['y']
    A = loadmat('Large/A.mat')['A']
    M,N,P = (49,49,49)
    D_x,D_y,D_z=construct_D_matrices_3D_sparse(M,N,P)
    L = sparse.vstack([D_x, D_y])
    L = sparse.vstack([L,D_z])
    lambda_reg = 1e-1
    x_cgls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=lambda_reg)

    print('CGLS done')
    x_0 = x_cgls
    alpha=0.5/2

    x_irls,res,k = cgls(A,L,y,max_iter=10000,lambda_reg=alpha,x_0=x_0,eps=1e-6,irls=True)

    X_irls = x_irls.reshape(M,N,P)
    fig = plt.figure(figsize=(20, 16))

    ax1 = fig.add_subplot(231, projection="3d")
    ax1.set_title(f"Small bag 3D plot of IRLS, alpha={alpha*2},Converged in {k} iterations")
    x, y, z = np.where(X_irls > 0.5)
    scatter=ax1.scatter(x, y, z, c=X_irls[x, y, z], cmap="viridis", marker="o")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label("Intensity")
    np.save('x_irls.npz',x_irls)
    plt.show()

Q16()
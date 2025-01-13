
def create_sphere(M, N, P, radius=0.5):
    """
    Create a 3D array with a spherical structure.
    
    Parameters:
    - M: int, number of rows in the 3D grid (y-axis)
    - N: int, number of columns in the 3D grid (x-axis)
    - P: int, number of slices in the 3D grid (z-axis)
    - radius: float, radius of the sphere
    
    Returns:
    - sphere: np.ndarray, a 3D array with values representing the sphere
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, M)
    z = np.linspace(-1, 1, P)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    # Create a sphere: 1 inside the sphere, 0 outside
    sphere = (X**2 + Y**2 + Z**2 <= radius**2).astype(float)
    return sphere


def plot_3d_subplots(X, Dx, Dy, Dz, threshold=0.1):
    """
    Plot the sphere, its gradients (Dx, Dy, Dz), and gradient magnitude in a single plot with 5 subplots.

    Parameters:
    - X: np.ndarray, the original sphere
    - Dx: np.ndarray, gradient along x-axis
    - Dy: np.ndarray, gradient along y-axis
    - Dz: np.ndarray, gradient along z-axis
    - threshold: float, minimum value to consider for visualization
    """
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(Dx**2 + Dy**2 + Dz**2)

    fig = plt.figure(figsize=(20, 16))

    # Original sphere
    ax1 = fig.add_subplot(231, projection="3d")
    ax1.set_title("Original Sphere")
    x, y, z = np.where(X > threshold)
    ax1.scatter(x, y, z, c=X[x, y, z], cmap="viridis", marker="o")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")

    # Gradient along x-axis
    ax2 = fig.add_subplot(232, projection="3d")
    ax2.set_title("Gradient along X-axis (Dx)")
    x, y, z = np.where(np.abs(Dx) > threshold)
    ax2.scatter(x, y, z, c=Dx[x, y, z], cmap="coolwarm", marker="o")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("Z-axis")

    # Gradient along y-axis
    ax3 = fig.add_subplot(233, projection="3d")
    ax3.set_title("Gradient along Y-axis (Dy)")
    x, y, z = np.where(np.abs(Dy) > threshold)
    ax3.scatter(x, y, z, c=Dy[x, y, z], cmap="coolwarm", marker="o")
    ax3.set_xlabel("X-axis")
    ax3.set_ylabel("Y-axis")
    ax3.set_zlabel("Z-axis")

    # Gradient along z-axis
    ax4 = fig.add_subplot(234, projection="3d")
    ax4.set_title("Gradient along Z-axis (Dz)")
    x, y, z = np.where(np.abs(Dz) > threshold)
    ax4.scatter(x, y, z, c=Dz[x, y, z], cmap="coolwarm", marker="o")
    ax4.set_xlabel("X-axis")
    ax4.set_ylabel("Y-axis")
    ax4.set_zlabel("Z-axis")

    # Gradient magnitude
    ax5 = fig.add_subplot(235, projection="3d")
    ax5.set_title("Gradient Magnitude (sqrt(Dx^2 + Dy^2 + Dz^2))")
    x, y, z = np.where(gradient_magnitude > threshold)
    ax5.scatter(x, y, z, c=gradient_magnitude[x, y, z], cmap="plasma", marker="o")
    ax5.set_xlabel("X-axis")
    ax5.set_ylabel("Y-axis")
    ax5.set_zlabel("Z-axis")

    plt.tight_layout()
    plt.savefig('3d_plots.png')


def check_3d():
    M, N, P = 19, 19, 19  # Grid dimensions
    X = create_sphere(M, N, P)  # Create a sphere
    D_x, D_y, D_z = construct_D_matrices_3D_sparse(M, N, P)  # Construct D matrices

    # Flatten the sphere and apply finite differences
    x = X.flatten('F')
    Dx = (D_x @ x).reshape((M, N, P), order="F")
    Dy = (D_y @ x).reshape((M, N, P), order="F")
    Dz = (D_z @ x).reshape((M, N, P), order="F")

    # Plot the original sphere and its gradients in 4 subplots
    plot_3d_subplots(X, Dx, Dy, Dz)





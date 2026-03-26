#%%
import vedo as vd
vd.settings.default_backend= 'vtk'

from vedo import show
import numpy as np

from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import triangle as tr # pip install triangle    

#%% Stencil class
# Abstract class for a stencil, which defines how elements are extracted from a mesh and how
# their variables are mapped between local and global degrees of freedom (DOFs).
# The stencil is responsible for:
# 1. Extracting elements from the mesh (e.g., edges, triangles)
# 2. Converting between array form and individual variables
# 3. Mapping local DOFs to global DOFs for assembly
# Each concrete stencil implementation must specify:
# - The dimension of the problem (2D or 3D)
# - The number of DOFs per element
# - How to extract elements from the mesh
# - How to convert between array and variable forms
# - How to map local DOFs to global DOFs

class Stencil(ABC):
    def __init__(self, dim=2, dof_per_vertex=2):
        self.dim = dim
        self.dof_per_vertex = dof_per_vertex

    @abstractmethod
    def create_elements(V,F):
        """
        Create elements from the mesh connectivity matrix F
        Args:
            F: Mesh connectivity matrix
        Returns:
            X: Undeformed coordinates of the element's vertices
            List of elements, where each element is a tuple of vertex indices
        """
        return 0

    @abstractmethod
    def to_variables(x): 
        """
        Convert from array form to individual variables
        Args:
            x: Array of variables for an element
        Returns:
            Tuple of individual variables
        """
        return 0

    @abstractmethod
    def local_to_global_dofs(self, element, nV):
        """
        Convert local DOF indices to global DOF indices
        Args:
            element: list of node indices for the element
            nV: number of vertices in the mesh
        Returns:
            list of global DOF indices
        """
        pass

class EdgeStencil(Stencil):
    """
    A stencil for edge elements in a mesh.
    An edge is defined by two vertices, and each vertex has 'dim' (2 or 3) degrees of freedom (DOFs).
    For example, in 2D each vertex has x and y coordinates, and in 3D each vertex has x, y, and z coordinates.
    
    The DOFs are ordered as [x1, y1, x2, y2] for 2D or [x1, y1, z1, x2, y2, z2] for 3D.
    """
    def __init__(self, dim=2):
        """
        Initialize the edge stencil.
        Args:
            dim: Dimension of the problem (2 for 2D, 3 for 3D)
        """
        super().__init__(dim, dof_per_vertex=2)  # Edge has 2 nodes
    
    @staticmethod
    def create_elements(V, F):
        """
        Create edge elements from a triangle mesh.
        For each triangle, creates its three edges, ensuring each edge is only included once
        regardless of how many triangles share it.
        
        Args:
            F: Triangle connectivity matrix of shape (n_triangles, 3)
        Returns:
            List of edges, where each edge is a sorted tuple of vertex indices
        """
        X = V.copy()
        edges = list({tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)})
        # Explanation for the line above:
        # For each triangle (i) and each of its three vertices (j):
        # 1. F[i,j] and F[i,(j+1)%3] get the indices of two consecutive vertices
        # 2. tuple(sorted(...)) creates a unique representation of the edge
        # 3. Using a set comprehension {} ensures each edge is only included once
        #    even if multiple triangles share the same edge
        return X, edges
    
    def to_variables(self, x):
        """
        Convert from array form to individual vertex coordinates.
        For an edge with vertices v1 and v2, the input array x should be of shape (2, dim).
        
        Args:
            x: Array of shape (2, dim) containing the coordinates of both vertices
        Returns:
            Tuple of two arrays, each containing the coordinates of one vertex
        """
        return x[0,:], x[1,:]

    def local_to_global_dofs(self, element, nV):
        """
        Convert local DOF indices to global DOF indices for an edge.
        For each vertex in the edge, maps its DOFs to the global system.
        
        Args:
            element: Tuple of two vertex indices defining the edge
            nV: Number of vertices in the mesh
        Returns:
            List of global DOF indices for the edge
        """
        # For an edge, each vertex has 'dim' degrees of freedom (DOFs) (x,y,z, etc.)
        # The DOFs are ordered as x1, y1, x2, y2, etc.
        global_dofs = []
        for node in element:
            # Add DOFs for this node
            global_dofs.extend([node*self.dim + d for d in range(self.dim)])
        return global_dofs

#%% Energy functions
# Abstract element energy class that defines the interface for energy functions.
# Each concrete energy implementation must specify:
# - The dimension of the problem (2D or 3D)
# - The energy function for a given element configuration
# - The gradient of the energy with respect to the element's DOFs
# - The hessian of the energy with respect to the element's DOFs
#
# The energy class is independent of how elements are stored or extracted from the mesh.
# It works with individual variables passed by the stencil, making it more modular and reusable.
# The exact number and type of variables needed is determined by each concrete implementation.

class ElementEnergy(ABC):    
    def __init__(self, dim=2):
        self.dim = dim

    @abstractmethod
    def energy(self, *args):
        """
        Compute the energy of an element in its current configuration.
        The exact arguments needed are determined by the concrete implementation.
        """
        return 0

    # should be overridden by the derived class, otherwise the finite difference implementation will be used
    def gradient(self, *args):
        """
        Compute the gradient of the energy with respect to the element's DOFs.
        The exact arguments needed are determined by the concrete implementation.
        """
        return self.gradient_fd(*args)

    def hessian(self, *args):
        """
        Compute the hessian of the energy with respect to the element's DOFs.
        The exact arguments needed are determined by the concrete implementation.
        """
        return self.hessian_fd(*args)
    
    # finite difference gradient and hessian
    def gradient_fd(self, *args):
        # TODO
        pass

    def hessian_fd(self, *args):
        # TODO
        pass
    
    # check that the gradient is correct by comparing it to the finite difference gradient
    def check_gradient(self, *args):
        grad = self.gradient(*args)
        grad_fd = self.gradient_fd(*args)
        return np.linalg.norm(grad - grad_fd)


# Spring energy function for a zero-length spring, defined as E = 0.5*||x1-x2||^2
# This energy penalizes the distance between two points, regardless of their rest configuration.
class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self, dim=2):
        super().__init__(dim)
    
    def energy(self, X1, X2, x1, x2):
        """
        Compute the zero-length spring energy.
        The energy is proportional to the squared distance between the points.
        Args:
            X1, X2: Undeformed coordinates of the element's vertices
            x1, x2: Deformed coordinates of the element's vertices
        Returns:
            Scalar energy value
        """
        return 0.5*np.linalg.norm(x1 - x2)**2

    def gradient(self, X1, X2, x1, x2):
        """
        Compute the gradient of the zero-length spring energy.
        The gradient is the difference between the points' positions.
        Args:
            X1, X2: Undeformed coordinates of the element's vertices
            x1, x2: Deformed coordinates of the element's vertices
        Returns:
            Gradient vector with respect to all DOFs
        """
        grad = np.zeros(2*self.dim)
        grad[:self.dim] = x1 - x2
        grad[self.dim:] = x2 - x1
        return grad

    def hessian(self, X1, X2, x1, x2):
        """
        Compute the hessian of the zero-length spring energy.
        The hessian is constant and has the form [I -I; -I I], where I is the identity matrix.
        Args:
            X1, X2: Undeformed coordinates of the element's vertices
            x1, x2: Deformed coordinates of the element's vertices
        Returns:
            Hessian matrix with respect to all DOFs
        """
        I = np.eye(self.dim)
        return np.block([[I, -I], [-I, I]])
    
# Spring energy function for a spring with a rest length, defined as E = 0.5*(||x1-x2|| - l)^2
# where l is the rest length (computed from the undeformed configuration).
class SpringEnergy(ElementEnergy):
    def __init__(self, dim=2):
        super().__init__(dim)
    
    def energy(self, X1, X2, x1, x2):
        """
        Compute the spring energy with rest length.
        The energy is proportional to the squared difference between the current length
        and the rest length (computed from the undeformed configuration).
        Args:
            X1, X2: Undeformed coordinates of the element's vertices
            x1, x2: Deformed coordinates of the element's vertices
        Returns:
            Scalar energy value
        """
        rest_length = np.linalg.norm(X1 - X2)
        return 0.5*(np.linalg.norm(x1-x2) - rest_length)**2


#%% Mesh class
class FEMMesh:
    """
    A class representing a finite element mesh.
    
    The mesh is defined by:
    - V: Vertex positions stored as a nV x dim matrix, where:
        * nV is the number of vertices
        * dim is the dimension of the problem (2 or 3)
        * Each row represents a vertex's coordinates
    - F: Face connectivity matrix defining the mesh topology
    - energy: The energy function to be minimized
    - stencil: The stencil defining how to extract elements from the mesh
    
    The class handles:
    - Computing the total energy of the mesh
    - Computing gradients and hessians for optimization
    - Assembling local element quantities into global system matrices
    """
    def __init__(self, V, F, energy, stencil):
        """
        Initialize the finite element mesh.
        
        Args:
            V: Vertex positions as a nV x dim matrix
            F: Face connectivity matrix
            energy: The energy function to be minimized
            stencil: The stencil defining how to extract elements
        """
        self.energy = energy
        self.stencil = stencil
        self.X, self.elements = self.stencil.create_elements(V,F)
        self.nV = self.X.shape[0]
        self.dim = self.energy.dim
        self.dof_per_vertex = self.stencil.dof_per_vertex

    def compute_energy(self,x):
        energy = 0
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            X_vars = self.stencil.to_variables(Xi)
            x_vars = self.stencil.to_variables(xi)
            energy += self.energy.energy(*X_vars, *x_vars)
        return energy
    
    def compute_local_gradients(self,x):
        gi=[]
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            X_vars = self.stencil.to_variables(Xi)
            x_vars = self.stencil.to_variables(xi)
            gi.append(self.energy.gradient(*X_vars, *x_vars))
        return gi

    def assemble_global_gradient(self,gi):
        grad = np.zeros(self.dim*self.nV)
        for i, element in enumerate(self.elements):
            global_dofs = self.stencil.local_to_global_dofs(element, self.nV)
            grad[global_dofs] += gi[i]
        return grad

    def compute_gradient(self,x):
        gi = self.compute_local_gradients(x)
        return self.assemble_global_gradient(gi)
    
    def compute_local_hessians(self,x):
        hi = []
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            X_vars = self.stencil.to_variables(Xi)
            x_vars = self.stencil.to_variables(xi)
            hi.append(self.energy.hessian(*X_vars, *x_vars))
        return hi

    def assemble_global_hessian(self,hi):
        # create arrays to store the sparse hessian matrix
        I = []
        J = []
        S = []
        for i, element in enumerate(self.elements):
            global_dofs = self.stencil.local_to_global_dofs(element, self.nV)
            for j, dof_i in enumerate(global_dofs):
                for k, dof_j in enumerate(global_dofs):
                    I.append(dof_i)
                    J.append(dof_j)
                    S.append(hi[i][j,k])
        H = coo_matrix((S, (I, J)), shape=(self.dim*self.nV, self.dim*self.nV))
        return H

    def compute_hessian(self,x):
        hi = self.compute_local_hessians(x)
        return self.assemble_global_hessian(hi)

            
#%% Optimization
class MeshOptimizer:
    """
    A class for optimizing mesh configurations using gradient-based methods.
    
    The optimizer supports different search directions (gradient descent, Newton) and
    uses backtracking line search to ensure convergence.
    
    The optimization process:
    1. Compute search direction (gradient descent or Newton)
    2. Perform line search to find step size
    3. Update mesh configuration
    4. Repeat until convergence
    
    The mesh configuration is represented as a flattened array of vertex positions,
    where each vertex has 'dim' coordinates (x,y for 2D, x,y,z for 3D).
    """
    def __init__(self, femMesh):
        """
        Initialize the mesh optimizer.
        
        Args:
            femMesh: The FEMMesh object to optimize
        """
        self.femMesh = femMesh
        self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1, max_iter=100):
        """
        Perform backtracking line search to find a step size that reduces the energy.
        
        Args:
            x: Current mesh configuration
            d: Search direction
            alpha: Initial step size
            max_iter: Maximum number of backtracking steps
        Returns:
            Tuple of (new configuration, step size)
        """
        x0 = x.copy()
        f0 = self.femMesh.compute_energy(x0)
        for _ in range(max_iter):
            if self.femMesh.compute_energy(x0 + alpha*d) <= f0:
                return x0 + alpha*d, alpha
            alpha *= 0.5
        return x0, alpha  # Return original point if no improvement found
    

    def GradientDescent(self, x):
        """
        Compute the gradient descent search direction.
        
        Args:
            x: Current mesh configuration
        Returns:
            Search direction (negative gradient)
        """
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        """
        Compute the Newton search direction by solving the linear system H*d = -g,
        where H is the hessian and g is the gradient.
        
        Args:
            x: Current mesh configuration
        Returns:
            Search direction (solution to H*d = -g)
        """
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)
        d = -spsolve(hess, grad) # solve the linear system hess * d = -grad
        return d
    
    def step(self, x):
        """
        Perform one optimization step.
        
        Args:
            x: Current mesh configuration
        Returns:
            New mesh configuration
        """
        d = self.SearchDirection(x)
        new_x, alpha = self.LineSearch(x,d)
        return new_x

    def optimize(self, x, max_iter=100, tol=1e-6):
        """
        Optimize the mesh configuration until convergence.
        
        Args:
            x: Initial mesh configuration
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (gradient norm)
        Returns:
            Optimized mesh configuration
        """
        for i in range(max_iter):
            x = self.step(x)
            if np.linalg.norm(self.femMesh.compute_gradient(x)) < tol:
                break
        return x

#%% Main program
# Create example meshes in 2D and 3D
# Vertices are stored as nV x dim matrices, where:
# - nV is the number of vertices
# - dim is the dimension (2 or 3)
# - Each row represents a vertex's coordinates

# 2D example (square)
vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
tris = tr.triangulate({"vertices":vertices}, f'qa0.01') # triangulate the square
V = tris['vertices'] # get the vertices of the triangles
F = tris['triangles'] # get the triangles


# Create 2D mesh with zero-length spring energy
energy = ZeroLengthSpringEnergy(dim=2)
mesh = FEMMesh(V, F, energy, EdgeStencil(dim=2))


pinned_vertices = []

def redraw():
    plt.remove("Mesh")
    mesh = vd.Mesh([V,F]).linecolor('black')
    plt.add(mesh)
    plt.remove("Points")
    plt.add(vd.Points(V[pinned_vertices,:],r=10))
    plt.render()

def OnLeftButtonPress(event):
    if event.object is None:          # mouse hits nothing, return.
        print('Mouse hits nothing')
    if isinstance(event.object,vd.mesh.Mesh):          # mouse hits the mesh
        Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
        print('Mouse hits the mesh')
        print('Coordinates:', event.picked3d)
        print('Point ID:', Vi)
        if Vi not in pinned_vertices:
            pinned_vertices.append(Vi)
        else:
            pinned_vertices.remove(Vi)
    redraw()

plt = vd.Plotter()

plt.add_callback('LeftButtonPress', OnLeftButtonPress) # add Keyboard callback
vdmesh = vd.Mesh([V,F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices,:])
plt.user_mode('2d').show().close()

# %%
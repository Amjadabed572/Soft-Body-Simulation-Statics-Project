#%%
import vedo as vd
vd.settings.default_backend={'vtk'}

from vedo import show, Mesh, Points
import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import triangle as tr  # pip install triangle

#%% Stencil classes
class Stencil(ABC):
    def __init__(self, dim=2, dof_per_vertex=2):
        self.dim = dim
        self.dof_per_vertex = dof_per_vertex
    @abstractmethod
    def create_elements(self, V, F): pass
    @abstractmethod
    def to_variables(self, x): pass
    @abstractmethod
    def local_to_global_dofs(self, element, nV): pass

class EdgeStencil(Stencil):
    def __init__(self, dim=2): super().__init__(dim, dof_per_vertex=dim)
    @staticmethod
    def create_elements(V, F):
        X = V.copy()
        edges = list({tuple(sorted((F[i,j], F[i,(j+1)%F.shape[1]])))
                      for i in range(F.shape[0]) for j in range(F.shape[1])})
        return X, edges
    def to_variables(self, x): return x[0,:], x[1,:]
    def local_to_global_dofs(self, element, nV):
        gd=[]
        for v in element: gd.extend([v*self.dim + d for d in range(self.dim)])
        return gd

#%% Energy functions (omitted for brevity)
# ... assume ElementEnergy, ZeroLengthSpringEnergy, SpringEnergy, FEMMesh, MeshOptimizer defined above ...

#%% Main program: Task 1
if __name__ == '__main__':
    # Task 1.1: Creative Shape (5-Point Star)
    n_pts = 10
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    R_outer, R_inner = 1.0, 0.4
    star_pts = np.array([
        [(R_outer if i % 2 == 0 else R_inner) * np.cos(a),
         (R_outer if i % 2 == 0 else R_inner) * np.sin(a)]
        for i, a in enumerate(angles)
    ])
    # Define closed segments for the star polygon
    segments_star = [(i, (i+1) % n_pts) for i in range(n_pts)]
    tri_star = tr.triangulate({'vertices': star_pts, 'segments': segments_star}, 'pzqa0.01')
    V_star, F_star = tri_star['vertices'], tri_star['triangles']
    plt_star = vd.Plotter(title='Task 1.1 – Star Mesh')
    plt_star += Mesh([V_star, F_star]).linecolor('black')
    plt_star.show().close()

    # Task 1.2: Circle Mesh (100 Boundary Vertices, No Interior Vertices)
    N = 100
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    # Define closed segments for the circle boundary
    segments_circle = [(i, (i+1) % N) for i in range(N)]
    tri_circle = tr.triangulate({'vertices': circle_pts, 'segments': segments_circle}, 'pzS')
    V_circle, F_circle = tri_circle['vertices'], tri_circle['triangles']
    plt_circle = vd.Plotter(title='Task 1.2 – Circle Mesh')
    plt_circle += Mesh([V_circle, F_circle]).linecolor('black')
    plt_circle.show().close()

    print('Task 1 complete: Star and Circle meshes created.')

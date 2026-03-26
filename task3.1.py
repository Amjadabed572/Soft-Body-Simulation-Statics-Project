import vedo as vd
import numpy as np
import triangle as tr

# ——— Zero-Length Spring Energy for a mesh ———
class ZeroLengthSpringEnergy:
    def __init__(self, V, F):
        self.n = V.shape[0]
        edges = set()
        for tri in F:
            for a, b in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])):
                edges.add(tuple(sorted((a, b))))
        self.edges = list(edges)

    def energy(self, x):
        pts = x.reshape((-1,3))
        E = 0.0
        for i, j in self.edges:
            v = pts[i] - pts[j]
            E += np.dot(v, v)
        return 0.5 * E

    def gradient(self, x):
        pts = x.reshape((-1,3))
        grad = np.zeros_like(pts)
        for i, j in self.edges:
            v = pts[i] - pts[j]
            grad[i] += v
            grad[j] -= v
        return grad.flatten()

    def hessian(self, x):
        dim = self.n * 3
        H = np.zeros((dim, dim))
        I3 = np.eye(3)
        for i, j in self.edges:
            H_blk = I3
            # contributions
            H[3*i:3*i+3, 3*i:3*i+3] += H_blk
            H[3*j:3*j+3, 3*j:3*j+3] += H_blk
            H[3*i:3*i+3, 3*j:3*j+3] -= H_blk
            H[3*j:3*j+3, 3*i:3*i+3] -= H_blk
        return H

# ——— Standard Hookean Spring Energy with analytic Hessian ———
class SpringEnergy:
    def __init__(self, V, F, k=1.0):
        self.n = V.shape[0]
        edges = set()
        for tri in F:
            for a, b in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])):
                edges.add(tuple(sorted((a, b))))
        self.edges = list(edges)
        self.rest = {edge: np.linalg.norm(V[edge[0]] - V[edge[1]]) for edge in self.edges}
        self.k = k

    def energy(self, x):
        pts = x.reshape((-1,3))
        E = 0.0
        for i, j in self.edges:
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            E += 0.5 * self.k * (L - self.rest[(i, j)])**2
        return E

    def gradient(self, x):
        pts = x.reshape((-1,3))
        grad = np.zeros_like(pts)
        for i, j in self.edges:
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L == 0:
                continue
            coeff = self.k * (L - self.rest[(i, j)]) / L
            d = coeff * v
            grad[i] += d
            grad[j] -= d
        return grad.flatten()

    def hessian(self, x):
        pts = x.reshape((-1,3))
        dim = self.n * 3
        H = np.zeros((dim, dim))
        I3 = np.eye(3)
        for i, j in self.edges:
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L == 0:
                continue
            # analytic Hessian block
            outer = np.outer(v, v)
            term1 = (1 - self.rest[(i, j)]/L) * I3
            term2 = (self.rest[(i, j)]/(L**3)) * outer
            H_blk = self.k * (term1 + term2)
            # add to global Hessian
            H[3*i:3*i+3, 3*i:3*i+3] += H_blk
            H[3*j:3*j+3, 3*j:3*j+3] += H_blk
            H[3*i:3*i+3, 3*j:3*j+3] -= H_blk
            H[3*j:3*j+3, 3*i:3*i+3] -= H_blk
        return H

# ——— FEMMesh wrapper ———
class FEMMesh:
    def __init__(self, mesh_data, energy_cls):
        V, F = mesh_data
        self.energy = energy_cls(V, F)

# ——— Optimizer using analytic gradient and Hessian ———
class MeshOptimizer:
    def __init__(self, fem_mesh, method='gradient', alpha=0.1, reg=1e-3):
        self.fem = fem_mesh
        self.method = method
        self.alpha = alpha
        self.reg = reg

    def step(self, x):
        # gradient
        g = self.fem.energy.gradient(x)
        if self.method == 'gradient':
            return x - self.alpha * g

        # Newton's method
        H = self.fem.energy.hessian(x)
        H_reg = H + self.reg * np.eye(x.size)
        dx = np.linalg.solve(H_reg, g)
        return x - self.alpha * dx

# ——— Star mesh generator ———
def make_star_mesh(n_pts=10, R_outer=1.0, R_inner=0.4, mesh_args='pzqa0.01'):
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    pts2d = np.array([[(R_outer if i%2==0 else R_inner) * np.cos(a),
                       (R_outer if i%2==0 else R_inner) * np.sin(a)]
                      for i, a in enumerate(angles)])
    tri = tr.triangulate({'vertices': pts2d,
                          'segments': [(i, (i+1)%n_pts) for i in range(n_pts)]},
                         mesh_args)
    V = np.column_stack([tri['vertices'], np.zeros(len(tri['vertices']))])
    F = tri['triangles']
    return V, F

# ——— Run optimization ———
def run_optimization(mesh_data, energy_cls, method='gradient', iters=5, perturb=False):
    V_orig, F = mesh_data
    if perturb and energy_cls is SpringEnergy:
        np.random.seed(42)
        V_init = V_orig + 0.1 * (np.random.rand(*V_orig.shape) - 0.5)
    else:
        V_init = V_orig.copy()
    fem = FEMMesh((V_init, F), energy_cls)
    optimizer = MeshOptimizer(fem, method=method, alpha=0.1, reg=1e-3)
    x = V_init.flatten()

    mesh_actor = vd.Mesh([V_init, F]).c('yellow').linecolor('black')
    plt = vd.Plotter(title=f"{energy_cls.__name__} + {method}", interactive=False)
    plt.add(mesh_actor)

    print(f"--- {energy_cls.__name__} ({method}) ---")
    for i in range(1, iters+1):
        x = optimizer.step(x)
        pts = x.reshape((-1,3))
        mesh_actor.points = pts
        plt.render()
        print(f"Iteration {i}: energy = {fem.energy.energy(x):.6f}")
    plt.show(axes=1)

# ——— Main ———
def main():
    mesh_data = make_star_mesh()
    run_optimization(mesh_data, ZeroLengthSpringEnergy, method='gradient')
    run_optimization(mesh_data, ZeroLengthSpringEnergy, method='newton')
    run_optimization(mesh_data, SpringEnergy, method='gradient', perturb=True)
    run_optimization(mesh_data, SpringEnergy, method='newton',    perturb=True)

if __name__ == '__main__':
    main()

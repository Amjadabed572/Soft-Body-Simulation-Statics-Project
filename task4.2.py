import vedo as vd
import numpy as np
import triangle as tr

# Mesh generator
def make_star_mesh(n=16, R_outer=1.0, R_inner=0.4, flags='pzqa0.01'):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts2d = np.array([
        [(R_outer if i % 2 == 0 else R_inner) * np.cos(a),
         (R_outer if i % 2 == 0 else R_inner) * np.sin(a)]
        for i, a in enumerate(angles)
    ])
    segments = [(i, (i+1) % n) for i in range(n)]
    tri = tr.triangulate({'vertices': pts2d, 'segments': segments}, flags)
    V2D = tri['vertices']
    F = tri['triangles']
    V = np.hstack([V2D, np.zeros((len(V2D), 1))])
        # raise mesh to start above sphere
    V[:, 2] += 1.0
    return V, F

# ——— Energy classes —————————————————————————————————————————
class ZeroLengthSpringEnergy:
    def __init__(self, V, F):
        self.edges = {tuple(sorted(e))
                      for tri in F
                      for e in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0]))}
    def energy(self, x):
        pts = x.reshape(-1,3)
        return sum(0.5 * np.dot(pts[i]-pts[j], pts[i]-pts[j]) for i,j in self.edges)
    def gradient(self, x):
        pts = x.reshape(-1,3)
        g = np.zeros_like(pts)
        for i,j in self.edges:
            diff = pts[i] - pts[j]
            g[i] += diff
            g[j] -= diff
        return g.ravel()

class SpringEnergy(ZeroLengthSpringEnergy):
    def __init__(self, V, F, k=10.0):
        super().__init__(V, F)
        self.k = k
        self.rest = {e: np.linalg.norm(V[e[0]] - V[e[1]]) for e in self.edges}
    def energy(self, x):
        pts = x.reshape(-1,3)
        return sum(0.5 * self.k * (np.linalg.norm(pts[i]-pts[j]) - L0)**2
                   for (i,j), L0 in self.rest.items())
    def gradient(self, x):
        pts = x.reshape(-1,3)
        g = np.zeros_like(pts)
        for (i,j), L0 in self.rest.items():
            diff = pts[i] - pts[j]
            L = np.linalg.norm(diff)
            if L > 1e-8:
                c = self.k * (L - L0) / L
                g[i] += c * diff
                g[j] -= c * diff
        return g.ravel()

class SoftConstraintEnergy:
    def __init__(self, base, pinned, targets, w=100.0):
        self.base = base
        self.pinned = pinned
        self.targets = np.array(targets)
        self.w = w
    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1,3)
        return E + sum(0.5 * self.w * np.dot(pts[i]-t, pts[i]-t)
                       for i, t in zip(self.pinned, self.targets))
    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1,3)
        pts = x.reshape(-1,3)
        for i, t in zip(self.pinned, self.targets):
            g[i] += self.w * (pts[i] - t)
        return g.ravel()

class GravityEnergy:
    """Uniform downward force"""
    def __init__(self, base, g=9.81):
        self.base = base
        self.g = g
    def energy(self, x):
        pts = x.reshape(-1,3)
        return self.base.energy(x) + self.g * np.sum(pts[:,2])
    def gradient(self, x):
        g0 = self.base.gradient(x).reshape(-1,3)
        # add gravity component in z
        g0[:,2] += self.g
        return g0.ravel()

class SpherePenaltyEnergy:
    def __init__(self, base, center, r, w=1e4):
        self.base = base
        self.center = np.array(center)
        self.r = r
        self.w = w
    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1,3)
        for p in pts:
            d = np.linalg.norm(p - self.center)
            if d < self.r:
                delta = self.r - d
                E += 0.5 * self.w * delta * delta
        return E
    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1,3)
        pts = x.reshape(-1,3)
        for i, p in enumerate(pts):
            diff = p - self.center
            d = np.linalg.norm(diff)
            if d < self.r:
                delta = self.r - d
                # push outwards
                g[i] -= self.w * delta * (diff / (d + 1e-10))
        return g.ravel()

class MeshOptimizer:
    def __init__(self, energy, alpha=5e-4):
        self.energy = energy
        self.alpha = alpha
    def step(self, x):
        return x - self.alpha * self.energy.gradient(x)

if __name__ == '__main__':
    # Create mesh
    V, F = make_star_mesh()
    # Pin two vertices at ground level
    pinned = [0, 8]
    center_z = V[:,2].mean()
    V0 = (V - np.array([0,0,center_z])) * 1.2 + np.array([0,0,center_z])
    targets = [V0[i].copy() for i in pinned]
    for t in targets:
        t[2] = 0.0

    # Build energy stack: springs, pins, gravity, sphere
    e_spring = SpringEnergy(V, F, k=10.0)
    e_pin    = SoftConstraintEnergy(e_spring, pinned, targets, w=100.0)
    e_grav   = GravityEnergy(e_pin, g=5.0)
    # Sphere as collider at (0,0,0), r=0.6
    e_sphere = SpherePenaltyEnergy(e_grav, center=[0,0,0], r=0.6, w=1e4)

    optimizer = MeshOptimizer(e_sphere, alpha=5e-4)
    x = V0.ravel()

    plt = vd.Plotter(bg='white', interactive=False)
    mesh_actor  = vd.Mesh((V0, F)).c('steelblue').alpha(0.6)
    sphere_actor = vd.Sphere(pos=(0,0,0), r=0.6).c('red').alpha(0.5)
    plt.show(mesh_actor, sphere_actor, 'Star Mesh Draped Over Sphere')

    # Relax
    for i in range(600):
        x = optimizer.step(x)
        mesh_actor.points = x.reshape(-1,3)
        if i % 80 == 0:
            print(f"Step {i}, Energy={e_sphere.energy(x):.4f}")
        plt.render()
    plt.close()

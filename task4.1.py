import vedo as vd
import numpy as np
import triangle as tr
import time

# — Mesh generator —
def make_star_mesh(n=16, R_outer=1.0, R_inner=0.4, flags='pzqa0.01'):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts2d = np.array([
        [ (R_outer if i % 2 == 0 else R_inner) * np.cos(a),
          (R_outer if i % 2 == 0 else R_inner) * np.sin(a) ]
        for i, a in enumerate(angles)
    ])
    segments = [(i, (i+1) % n) for i in range(n)]
    tri = tr.triangulate({'vertices': pts2d, 'segments': segments}, flags)
    verts2d = tri['vertices']
    triangles = tri['triangles']
    V = np.hstack([verts2d, np.zeros((len(verts2d), 1))])
    V[:, 2] -= 1.0
    return V, triangles

# — Energy classes —
class ZeroLengthSpringEnergy:
    def __init__(self, V, F):
        self.n = len(V)
        self.edges = set()
        for tri in F:
            e0 = tuple(sorted((tri[0], tri[1])))
            e1 = tuple(sorted((tri[1], tri[2])))
            e2 = tuple(sorted((tri[2], tri[0])))
            self.edges.add(e0)
            self.edges.add(e1)
            self.edges.add(e2)

    def energy(self, x):
        pts = x.reshape(-1, 3)
        E = 0.0
        for i, j in self.edges:
            diff = pts[i] - pts[j]
            E += 0.5 * np.dot(diff, diff)
        return E

    def gradient(self, x):
        pts = x.reshape(-1, 3)
        grad = np.zeros_like(pts)
        for i, j in self.edges:
            v = pts[i] - pts[j]
            grad[i] += v
            grad[j] -= v
        return grad.ravel()

    def hessian(self, x):
        dim = self.n * 3
        H = np.zeros((dim, dim))
        I3 = np.eye(3)
        for i, j in self.edges:
            H[3*i:3*i+3, 3*i:3*i+3] += I3
            H[3*j:3*j+3, 3*j:3*j+3] += I3
            H[3*i:3*i+3, 3*j:3*j+3] -= I3
            H[3*j:3*j+3, 3*i:3*i+3] -= I3
        return H

class SpringEnergy(ZeroLengthSpringEnergy):
    def __init__(self, V, F, k=1.0, rest_length_scale=1.0):
        super().__init__(V, F)
        self.k = k
        self.rest_lengths = {}
        for e in self.edges:
            i, j = e
            self.rest_lengths[e] = np.linalg.norm(V[i] - V[j]) * rest_length_scale

    def energy(self, x):
        pts = x.reshape(-1, 3)
        E = 0.0
        for e, L0 in self.rest_lengths.items():
            i, j = e
            curL = np.linalg.norm(pts[i] - pts[j])
            E += 0.5 * self.k * (curL - L0)**2
        return E

    def gradient(self, x):
        pts = x.reshape(-1, 3)
        grad = np.zeros_like(pts)
        for e, L0 in self.rest_lengths.items():
            i, j = e
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L > 1e-8:
                coeff = self.k * (L - L0) / (L + 1e-10)
                grad[i] += coeff * v
                grad[j] -= coeff * v
        return grad.ravel()

    def hessian(self, x):
        pts = x.reshape(-1, 3)
        dim = self.n * 3
        H = np.zeros((dim, dim))
        I3 = np.eye(3)
        for e, L0 in self.rest_lengths.items():
            i, j = e
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L > 1e-8:
                outer = np.outer(v, v)
                term1 = (1 - L0/(L + 1e-10)) * I3
                term2 = (L0/((L + 1e-10)**3)) * outer
                B = self.k * (term1 + term2)
                for (a, b, sign) in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
                    H[3*a:3*a+3, 3*b:3*b+3] += sign * B
        return H

class SoftConstraintEnergy:
    def __init__(self, base, pinned, targets, weight):
        self.base = base
        self.pinned = pinned
        self.targets = np.array(targets)
        self.weight = weight

    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1, 3)
        for idx, t in zip(self.pinned, self.targets):
            diff = pts[idx] - t
            E += 0.5 * self.weight * np.dot(diff, diff)
        return E

    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1, 3)
        pts = x.reshape(-1, 3)
        for idx, t in zip(self.pinned, self.targets):
            g[idx] += self.weight * (pts[idx] - t)
        return g.ravel()

    def hessian(self, x):
        H = self.base.hessian(x)
        for idx in self.pinned:
            i = 3*idx
            H[i:i+3, i:i+3] += self.weight * np.eye(3)
        return H

class GroundCollisionEnergy:
    def __init__(self, base, ground_z=0.0, weight=1000.0):
        self.base = base
        self.ground_z = ground_z
        self.weight = weight
        self.n = self.base.n

    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1, 3)
        for z in pts[:, 2]:
            if z < self.ground_z:
                d = self.ground_z - z
                E += 0.5 * self.weight * d * d
        return E

    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1, 3)
        pts = x.reshape(-1, 3)
        for i, z in enumerate(pts[:, 2]):
            if z < self.ground_z:
                diff = self.ground_z - z
                g[i, 2] -= self.weight * diff
        return g.ravel()

    def hessian(self, x):
        H = self.base.hessian(x)
        pts = x.reshape(-1, 3)
        for i, z in enumerate(pts[:, 2]):
            if z < self.ground_z:
                idx = 3*i + 2
                H[idx, idx] += self.weight
        return H

class RadialCompressionEnergy:
    def __init__(self, base, weight=100.0):
        self.base = base
        self.weight = weight
        self.n = self.base.n

    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1, 3)
        for xy in pts[:, :2]:
            r = np.linalg.norm(xy)
            E += 0.5 * self.weight * r * r
        return E

    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1, 3)
        pts = x.reshape(-1, 3)
        for i, xy in enumerate(pts[:, :2]):
            g[i, :2] += self.weight * xy
        return g.ravel()

    def hessian(self, x):
        H = self.base.hessian(x)
        dim = self.n * 3
        for i in range(self.n):
            idx = 3*i
            H[idx:idx+2, idx:idx+2] += self.weight * np.eye(2)
        return H

class MeshOptimizer:
    def __init__(self, energy, method='newton'):
        self.energy = energy
        self.method = method

    def step(self, x, alpha=0.1):
        g = self.energy.gradient(x)
        g = np.clip(g, -1.0, 1.0)
        if self.method == 'gradient':
            return x - alpha * g
        H = self.energy.hessian(x) + 1.0 * np.eye(len(g))
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dx = g
        E0 = self.energy.energy(x)
        for factor in [1, 0.5, 0.1, 0.01]:
            x_new = x - alpha * factor * dx
            if self.energy.energy(x_new) <= E0:
                return x_new
        return x - alpha * g

# — Relaxation runner —
def run_relax(V, F, pinned, targets, spring_k, pin_w, ground_z, ground_w, radial_w, rest_length_scale, scale, title):
    center = V.mean(axis=0)
    V0 = (V - center) * scale + center
    base_energy = SpringEnergy(V, F, k=spring_k, rest_length_scale=rest_length_scale)
    if ground_w > 0:
        base_energy = GroundCollisionEnergy(base_energy, ground_z, ground_w)
    if radial_w > 0:
        base_energy = RadialCompressionEnergy(base_energy, radial_w)
    total_energy = SoftConstraintEnergy(base_energy, pinned, targets, pin_w)
    optimizer = MeshOptimizer(total_energy)

    x = V0.ravel()
    plt = vd.Plotter(bg='white', interactive=False)
    mesh_actor = vd.Mesh((V0, F)).linewidth(1).c('steelblue').alpha(0.6)
    ground_plane = vd.Plane(pos=(0,0,ground_z), normal=(0,0,1), s=(3,3))
    ground_plane.c('lightgray').alpha(0.5)
    plt.show(mesh_actor, ground_plane, title, at=0)

    print(f"{title} - Initial max radius: {np.max(np.linalg.norm(V0[:, :2], axis=1)):.4f}")
    for i in range(100):
        x = optimizer.step(x, alpha=0.1)
        mesh_actor.points = x.reshape(-1, 3)
        if i % 20 == 0:
            pts = x.reshape(-1, 3)
            min_z = np.min(pts[:, 2])
            max_radius = np.max(np.linalg.norm(pts[:, :2], axis=1))
            energy = total_energy.energy(x)
            print(f"{title} - Step {i}: Min z = {min_z:.4f}, Max radius = {max_radius:.4f}, Energy = {energy:.4f}")
        plt.render()
        time.sleep(0.01)
    final_pts = x.reshape(-1, 3)
    final_radius = np.max(np.linalg.norm(final_pts[:, :2], axis=1))
    print(f"{title} - Final max radius: {final_radius:.4f}")
    plt.close()

# — Main demonstration —
def main():
    V, F = make_star_mesh(n=16)
    pinned = [0, 8]  # Pin two vertices
    # Springs Only: pin at z=-1.0
    targets_springs = [V[i].copy() for i in pinned]
    # With Collision: pin at z=0.0
    targets_collision = [V[i].copy() for i in pinned]
    for t in targets_collision:
        t[2] = 0.0
    print("Pinned vertices:", pinned)
    params = dict(
        spring_k=100.0,  # Stronger springs
        ground_z=0.0
    )
    scale_factor = 1.2

    # Without ground collision
    run_relax(V, F, pinned, targets_springs, pin_w=10.0, ground_w=0.0, radial_w=0.0, rest_length_scale=1.0,
              scale=scale_factor, title='Relax: Springs Only', **params)
    # With ground collision and radial compression
    run_relax(V, F, pinned, targets_collision, pin_w=50.0, ground_w=10000.0, radial_w=20.0, rest_length_scale=0.9,
              scale=scale_factor, title='Relax: With Collision', **params)

if __name__ == '__main__':
    main()
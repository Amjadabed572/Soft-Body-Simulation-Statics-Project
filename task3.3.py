import vedo as vd
import numpy as np
import triangle as tr

# ——— Energy classes —————————————————————————————————————————————————
class ZeroLengthSpringEnergy:
    def __init__(self, V, F):
        self.n = len(V)
        self.edges = {tuple(sorted(e)) for tri in F for e in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]}
    def energy(self, x):
        pts = x.reshape(-1,3)
        return 0.5 * sum(np.dot(pts[i]-pts[j], pts[i]-pts[j]) for i,j in self.edges)
    def gradient(self, x):
        pts = x.reshape(-1,3)
        grad = np.zeros_like(pts)
        for i,j in self.edges:
            v = pts[i] - pts[j]
            grad[i] += v
            grad[j] -= v
        return grad.ravel()
    def hessian(self, x):
        dim = self.n * 3
        H = np.zeros((dim, dim))  # Fixed syntax error
        I3 = np.eye(3)
        for i,j in self.edges:
            H[3*i:3*i+3, 3*i:3*i+3] += I3
            H[3*j:3*j+3, 3*j:3*j+3] += I3
            H[3*i:3*i+3, 3*j:3*j+3] -= I3
            H[3*j:3*j+3, 3*i:3*i+3] -= I3
        return H

class SpringEnergy(ZeroLengthSpringEnergy):
    def __init__(self, V, F, k=1.0):
        super().__init__(V, F)
        self.k = k
        self.rest = {e: np.linalg.norm(V[e[0]]-V[e[1]]) for e in self.edges}
    def energy(self, x):
        pts = x.reshape(-1,3)
        return 0.5 * sum(
            self.k * (np.linalg.norm(pts[i]-pts[j]) - self.rest[(i,j)])**2
            for i,j in self.edges
        )
    def gradient(self, x):
        pts = x.reshape(-1,3)
        grad = np.zeros_like(pts)
        for i,j in self.edges:
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L > 1e-8:
                d = (self.k * (L - self.rest[(i,j)]) / L) * v
                grad[i] += d
                grad[j] -= d
        return grad.ravel()
    def hessian(self, x):
        pts = x.reshape(-1,3)
        dim = len(pts) * 3
        H = np.zeros((dim, dim))
        I3 = np.eye(3)
        for i,j in self.edges:
            v = pts[i] - pts[j]
            L = np.linalg.norm(v)
            if L > 1e-8:
                outer = np.outer(v, v)
                term1 = (1 - self.rest[(i,j)]/L) * I3
                term2 = (self.rest[(i,j)]/(L**3)) * outer
                B = self.k * (term1 + term2)
                H[3*i:3*i+3, 3*i:3*i+3] += B
                H[3*j:3*j+3, 3*j:3*j+3] += B
                H[3*i:3*i+3, 3*j:3*j+3] -= B
                H[3*j:3*j+3, 3*i:3*i+3] -= B
        return H

class SoftConstraintEnergy:
    def __init__(self, base, pinned, targets, w):
        self.base = base
        self.pinned = pinned
        self.targets = np.array(targets)
        self.w = w
    def energy(self, x):
        E = self.base.energy(x)
        pts = x.reshape(-1,3)
        for i,t in zip(self.pinned, self.targets):
            d = pts[i] - t
            E += 0.5 * self.w * np.dot(d, d)
        return E
    def gradient(self, x):
        g = self.base.gradient(x).reshape(-1,3)
        pts = x.reshape(-1,3)
        for i,t in zip(self.pinned, self.targets):
            g[i] += self.w * (pts[i] - t)
        return g.ravel()
    def hessian(self, x):
        H = self.base.hessian(x)
        for i in self.pinned:
            idx = 3 * i
            H[idx:idx+3, idx:idx+3] += self.w * np.eye(3)
        return H

class MeshOptimizer:
    def __init__(self, energy, method='newton'):
        self.e = energy
        self.m = method
    def step(self, x, alpha=1.0):
        g = self.e.gradient(x)
        print(f"Gradient norm: {np.linalg.norm(g)}")
        if self.m == 'gradient':
            return x - alpha * g
        H = self.e.hessian(x) + 1e-3 * np.eye(len(g))
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Hessian solve failed, using gradient step")
            dx = g
        E0 = self.e.energy(x)
        for f in (1, 0.5, 0.1, 0.01):
            xn = x - alpha * f * dx
            E1 = self.e.energy(xn)
            print(f"Energy: {E0} -> {E1}, step factor: {f}")
            if E1 < E0:
                return xn
        print("No energy decrease, using gradient step")
        return x - alpha * g

# ——— Mesh creation —————————————————————————————————————————————————
def make_star_mesh(n=10):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    R_outer, R_inner = 1.0, 0.4
    pts2d = np.array([
        [(R_outer if i%2==0 else R_inner) * np.cos(a),
         (R_outer if i%2==0 else R_inner) * np.sin(a)]
        for i,a in enumerate(angles)
    ])
    segments = [(i, (i+1) % n) for i in range(n)]
    tri_data = {'vertices': pts2d, 'segments': segments}
    tri_out = tr.triangulate(tri_data, 'pzqa0.01')
    V = np.column_stack([tri_out['vertices'], np.zeros(len(tri_out['vertices']))])
    F = tri_out['triangles']
    return V, F

# ——— Main application ——————————————————————————————————————————————
if __name__ == '__main__':
    V, F = make_star_mesh()
    x = V.copy().ravel()
    pinned, targets = [], []
    weight = 20.0

    # Create plotter
    plt = vd.Plotter(title='Soft-Pinned Star', size=(800,600))

    mesh = vd.Mesh((V, F)).c('lightblue').linecolor('black')
    pts_act = vd.Points([], r=12, c='red')
    lbl = vd.Text2D(
        'Click mesh to pin/unpin | +/- to adjust weight | G/N step',
        pos='top-left', bg='white'
    )
    wlbl = vd.Text2D(f'Weight = {weight:.2f}', pos='top-right', bg='white')

    plt.add(mesh, pts_act, lbl, wlbl)

    def on_left_click(evt):
        global pts_act
        if evt.actor is not mesh:
            return
        click_pos = evt.picked3d
        if click_pos is None:
            return
        click_xy = np.array(click_pos)[:2]
        all_xy = x.reshape(-1, 3)[:, :2]
        idx = int(np.argmin(np.linalg.norm(all_xy - click_xy, axis=1)))
        if idx in pinned:
            i = pinned.index(idx)
            pinned.pop(i)
            targets.pop(i)
            print(f"Unpinned point {idx}")
        else:
            pinned.append(idx)
            target_pos = x.reshape(-1, 3)[idx].copy() + np.array([0.5, 0.5, 0])
            targets.append(target_pos)
            print(f"Pinned point {idx} at target {target_pos}")
        current_positions = x.reshape(-1, 3)[pinned] if pinned else []
        plt.remove(pts_act)
        pts_act = vd.Points(np.array(current_positions), r=12, c='red')
        plt.add(pts_act)
        plt.render()

    def on_key(evt):
        global weight, mesh
        key = evt.keypress.lower() if hasattr(evt, 'keypress') else None
        if not key:
            return

        if key == '+':
            weight = min(100.0, weight + 5.0)
            print(f"Weight increased to {weight}")
        elif key == '-':
            weight = max(0.0, weight - 5.0)
            print(f"Weight decreased to {weight}")
        elif key in ('g', 'n'):
            print(f"Applying {'gradient' if key == 'g' else 'Newton'} step")
            print(f"Pinned: {pinned}, Targets: {targets}")
            base = SpringEnergy(V, F, k=5.0)
            e = SoftConstraintEnergy(base, pinned, targets, w=weight)
            print(f"Initial energy: {e.energy(x)}")
            opt = MeshOptimizer(e, 'gradient' if key == 'g' else 'newton')
            x_old = x.copy()
            x[:] = opt.step(x)
            print(f"Vertex change norm: {np.linalg.norm(x - x_old)}")
            plt.remove(mesh)
            mesh = vd.Mesh((x.reshape(-1, 3), F)).c('lightblue').linecolor('black')
            plt.add(mesh)

        wlbl.text(f'Weight = {weight:.2f}')
        plt.render()

    # Test energy model
    print("Testing energy model...")
    test_pinned = [0]
    test_targets = [V[0] + np.array([0.5, 0.5, 0])]
    test_base = SpringEnergy(V, F, k=5.0)
    test_e = SoftConstraintEnergy(test_base, test_pinned, test_targets, w=20.0)
    print(f"Test energy: {test_e.energy(x)}")
    test_g = test_e.gradient(x)
    print(f"Test gradient norm: {np.linalg.norm(test_g)}")
    test_opt = MeshOptimizer(test_e, 'newton')
    x_test = x.copy()
    x_test[:] = test_opt.step(x_test)
    print(f"Test vertex change norm: {np.linalg.norm(x_test - x)}")

    plt.add_callback('LeftButtonPress', on_left_click)
    plt.add_callback('KeyPress', on_key)

    plt.show(axes=1)
import vedo as vd
vd.settings.default_backend = 'vtk'

from vedo import Mesh, Points, Text2D
import numpy as np
import triangle as tr


def main():
    # Generate the star mesh data
    n_pts = 10
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    R_outer, R_inner = 1.0, 0.4
    pts2d = np.array([
        [(R_outer if i % 2 == 0 else R_inner) * np.cos(a),
         (R_outer if i % 2 == 0 else R_inner) * np.sin(a)]
        for i, a in enumerate(angles)
    ])
    # Triangulate the 2D star
    tri_star = tr.triangulate(
        {'vertices': pts2d,
         'segments': [(i, (i + 1) % n_pts) for i in range(n_pts)]},
        'pzqa0.01'
    )
    pts2d = tri_star['vertices']
    V_star = np.column_stack([pts2d, np.zeros(len(pts2d))])
    F_star = tri_star['triangles']

    # Initialize the Plotter
    plt_ui = vd.Plotter(title='Task 2 – Pin & Move Vertices UI')
    msg = Text2D('Press 2 for 2D, 3 for 3D. Left-click to pin, right-press to drag.',
                  pos='top-left', font='Arial', bg='lightgray')
    plt_ui.add(msg)

    # Create initial actors
    mesh_actor = Mesh([V_star, F_star]).c('yellow').linecolor('black')
    pts_actor = Points(np.empty((0,3)), r=10, c='red')
    plt_ui.add(mesh_actor)
    plt_ui.add(pts_actor)

    pinned = []
    dragging = False
    drag_idx = None

    # Utility to refresh pin actor
    def refresh_pins():
        nonlocal pts_actor
        plt_ui.remove(pts_actor)
        if pinned:
            pts_actor = Points(V_star[pinned], r=10, c='red')
        else:
            pts_actor = Points(np.empty((0,3)), r=10, c='red')
        plt_ui.add(pts_actor)
        plt_ui.render()

    # Left-click: toggle pin/unpin
    def OnLeftClick(evt):
        nonlocal pinned
        if evt.picked3d is None or evt.actor not in (mesh_actor, pts_actor):
            return
        # find closest vertex in XY
        click2d = np.array(evt.picked3d)[:2]
        vid = int(np.argmin(np.linalg.norm(V_star[:,:2] - click2d, axis=1)))
        if vid not in pinned:
            pinned.append(vid)
            msg.text(f'Pinned vertex {vid}')
        else:
            pinned.remove(vid)
            msg.text(f'Unpinned vertex {vid}')
        refresh_pins()

    # Right-press: start dragging a pinned vertex
    def OnRightButtonPress(evt):
        nonlocal dragging, drag_idx
        if evt.picked3d is None or evt.actor not in (mesh_actor, pts_actor):
            return
        click2d = np.array(evt.picked3d)[:2]
        # pick nearest pinned
        if pinned:
            dists = np.linalg.norm(V_star[pinned,:2] - click2d, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < 0.2:
                dragging = True
                drag_idx = pinned[idx]
                msg.text(f'Start dragging {drag_idx}')
                plt_ui.render()

    # Right-release: stop dragging
    def OnRightButtonRelease(evt):
        nonlocal dragging, drag_idx
        if dragging:
            dragging = False
            msg.text(f'Dropped vertex {drag_idx}')
            drag_idx = None
            plt_ui.render()

    # MouseMove: update dragging
    def OnMouseMove(evt):
        nonlocal dragging
        if not dragging or drag_idx is None:
            return
        # compute new world point under mouse
        if evt.picked3d is None:
            return
        new3d = np.array(evt.picked3d)
        V_star[drag_idx] = new3d
        # update mesh geometry
        mesh_actor.points = V_star
        refresh_pins()
        msg.text(f'Moving {drag_idx} -> {new3d.round(2)}')

    # KeyPress: switch camera modes
    def OnKeyPress(evt):
        key = getattr(evt, 'key', '').lower()
        cam = plt_ui.camera
        if key == '2':
            cam.SetParallelProjection(True)
            cam.SetPosition(0,0,5)
            cam.SetFocalPoint(0,0,0)
            cam.SetViewUp(0,1,0)
            msg.text('Switched to 2D mode')
        elif key == '3':
            cam.SetParallelProjection(False)
            plt_ui.reset_camera()
            msg.text('Switched to 3D mode')
        plt_ui.render()

    # Register callbacks
    plt_ui.add_callback('LeftButtonPress', OnLeftClick)
    plt_ui.add_callback('RightButtonPress', OnRightButtonPress)
    plt_ui.add_callback('RightButtonRelease', OnRightButtonRelease)
    plt_ui.add_callback('MouseMove', OnMouseMove)
    plt_ui.add_callback('KeyPress', OnKeyPress)

    plt_ui.show(axes=1, interactive=True)

if __name__ == '__main__':
    main()

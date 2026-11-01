"""Surface extraction and rendering for stress contour plots (no FE dependency)."""
import os

import numpy as np

_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def boundary_triangles(tets):
    """Faces that belong to exactly one tetrahedron. `tets` holds the 4 corner node ids per cell."""
    faces = np.vstack([tets[:, f] for f in _TET_FACES])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def nodal_average(n_nodes, node_of_value, values):
    """Average values that belong to the same node (e.g. DG1 values of all cells at a vertex)."""
    total = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    np.add.at(total, node_of_value, values)
    np.add.at(count, node_of_value, 1.0)
    return total / np.maximum(count, 1.0)


def render_matplotlib(points, tris, values, clim, marker, label, title, out, n_levels=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    norm = colors.Normalize(*clim)
    cmap = plt.get_cmap("jet", n_levels)
    face_rgba = cmap(norm(values[tris].mean(axis=1)))
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(points[tris], facecolors=face_rgba,
                                         edgecolors=face_rgba, linewidths=0.3))
    used = points[np.unique(tris)]
    lo, hi = used.min(axis=0), used.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=22, azim=38)
    ax.set_axis_off()
    if marker is not None:                      # 2D annotation, so it stays on top of the surface
        from mpl_toolkits.mplot3d import proj3d
        x2, y2, _ = proj3d.proj_transform(*marker, ax.get_proj())
        ax.annotate(label, xy=(x2, y2), xytext=(-70, 60), textcoords="offset points",
                    fontsize=12, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="von Mises stress [MPa]")
    ax.set_title(title, fontsize=12)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_pyvista(points, tris, values, clim, marker, label, title, out, n_levels=10):
    import pyvista as pv

    pv.OFF_SCREEN = True
    cells = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64).ravel()
    surf = pv.PolyData(points, cells)
    surf.point_data["von Mises [MPa]"] = values
    pl = pv.Plotter(off_screen=True, window_size=(1700, 1100))
    pl.set_background("white")
    pl.add_mesh(surf, scalars="von Mises [MPa]", cmap="jet", clim=clim, n_colors=n_levels,
                lighting=False, show_edges=False,
                scalar_bar_args=dict(title="von Mises stress [MPa]", n_labels=n_levels + 1,
                                     fmt="%.0f", vertical=True, color="black",
                                     title_font_size=20, label_font_size=18))
    if marker is not None:
        pl.add_point_labels(np.array([marker]), [label], point_size=16, font_size=22,
                            text_color="black", point_color="black", shape_color="white",
                            shape_opacity=0.85, always_visible=True,
                            render_points_as_spheres=True)
    pl.add_text(title, position="upper_edge", font_size=14, color="black")
    pl.add_axes()
    pl.camera_position = "iso"
    pl.camera.zoom(1.15)
    pl.screenshot(out)
    pl.close()


def render(points, tris, values, clim, marker, label, title, out, renderer="auto"):
    """renderer: 'pyvista', 'matplotlib' or 'auto' (PyVista if a display exists, else matplotlib)."""
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if renderer == "pyvista" or (renderer == "auto" and has_display):
        try:
            render_pyvista(points, tris, values, clim, marker, label, title, out)
            return "pyvista"
        except Exception as e:                       # fall back instead of failing
            if renderer == "pyvista":
                raise
            print(f"PyVista rendering failed ({type(e).__name__}: {e}); using matplotlib")
    render_matplotlib(points, tris, values, clim, marker, label, title, out)
    return "matplotlib"
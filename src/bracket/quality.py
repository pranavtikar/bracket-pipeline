import numpy as np
import gmsh

_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _corner_connectivity():
    """Return (n_tets, 4) array of corner-node tags (works for linear or quadratic tets)."""
    types, _, nodes = gmsh.model.mesh.getElements(3)
    conns = []
    for et, nn in zip(types, nodes):
        npe = {4: 4, 11: 10}.get(int(et))
        if npe is None:
            raise ValueError(f"unsupported element type {et}")
        conns.append(np.asarray(nn, dtype=np.int64).reshape(-1, npe)[:, :4])
    return np.vstack(conns)


def quality_metrics():
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    xyz = np.zeros((ntags.max() + 1, 3))
    xyz[ntags] = np.asarray(coords).reshape(-1, 3)

    P = xyz[_corner_connectivity()]                       # (n, 4, 3)
    lens = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in _EDGES], axis=1)
    aspect = lens.max(axis=1) / lens.min(axis=1)

    a, b, c = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]
    triple = np.einsum("ij,ij->i", a, np.cross(b, c))
    V = np.abs(triple) / 6.0
    num = (np.sum(a * a, 1)[:, None] * np.cross(b, c)
           + np.sum(b * b, 1)[:, None] * np.cross(c, a)
           + np.sum(c * c, 1)[:, None] * np.cross(a, b))
    R = np.linalg.norm(num, axis=1) / (12.0 * V)           # circumradius
    edge_ideal = 4.0 * R / np.sqrt(6.0)
    V_ideal = edge_ideal ** 3 / (6.0 * np.sqrt(2.0))
    skew = np.clip(1.0 - V / V_ideal, 0.0, 1.0)

    return {
        "n_tets": len(V),
        "aspect_max": float(aspect.max()), "aspect_mean": float(aspect.mean()),
        "skew_max": float(skew.max()), "skew_mean": float(skew.mean()),
        "min_volume": float(V.min()),
    }
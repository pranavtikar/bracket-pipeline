import gmsh

L, W = 80.0, 40.0  # base length and width (mm), fixed


def is_valid(p):
    t, H, rh, rf = p["t"], p["H"], p["r_hole"], p["r_fillet"]
    return (
        rf < 0.5 * H
        and rf < L - t - 5
        and 2 * rh < 0.6 * W
        and 0.7 * H - rh - rf > 2.0   # hole clears the fillet
        and 0.3 * H - rh > 3.0        # hole clears the top of the leg
    )


def build_mesh(p, h_global, h_local, order=2):
    """Build the L-bracket in the current gmsh model and mesh it."""
    if not is_valid(p):
        raise ValueError(f"invalid geometry: {p}")
    occ = gmsh.model.occ
    t, H, rh, rf = p["t"], p["H"], p["r_hole"], p["r_fillet"]
    eps = 1e-3

    base = occ.addBox(0, 0, 0, L, W, t)
    leg = occ.addBox(0, 0, t, t, W, H)
    vol, _ = occ.fuse([(3, base)], [(3, leg)])
    occ.synchronize()

    # inner corner edge: runs along y at x = t, z = t
    edges = gmsh.model.getEntitiesInBoundingBox(
        t - eps, -eps, t - eps, t + eps, W + eps, t + eps, dim=1)
    if not edges:
        raise RuntimeError("inner corner edge not found")
    vol = occ.fillet([vol[0][1]], [e[1] for e in edges], [rf])  # returns one list
    occ.synchronize()

    # hole through the leg, axis along x
    cyl = occ.addCylinder(-1, W / 2, t + 0.7 * H, t + 2, 0, 0, rh)
    vol, _ = occ.cut(vol, [(3, cyl)])
    occ.synchronize()

    # physical groups: 1 = fixed (base bottom), 2 = loaded (leg top), 3 = body
    fixed = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, L + eps, W + eps, eps, dim=2)
    load = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, t + H - eps, t + eps, W + eps, t + H + eps, dim=2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in fixed], 1)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in load], 2)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], 3)

    # local refinement near all cylindrical surfaces (hole + fillet)
    crit = [s[1] for s in gmsh.model.getEntities(2)
            if gmsh.model.getType(2, s[1]) == "Cylinder"]
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    f.setNumbers(d, "SurfacesList", crit)
    f.setNumber(d, "Sampling", 50)
    th = f.add("Threshold")
    f.setNumber(th, "InField", d)
    f.setNumber(th, "SizeMin", h_local)
    f.setNumber(th, "SizeMax", h_global)
    f.setNumber(th, "DistMin", 1.0)
    f.setNumber(th, "DistMax", 8.0)
    f.setAsBackgroundMesh(th)
    for o in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{o}", 0)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    return crit
"""Linear-elastic FE solve of the L-bracket. Units: mm, N, MPa. Serial runs only."""
import time
from dataclasses import dataclass

import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
from bracket.geometry import HOLE_FRAC

E, NU = 210e3, 0.3      # steel
LOAD_N = 500.0          # total force on the leg top, in +x (N)


@dataclass
class FEAResult:
    ndof: int
    max_vm_roi: float       # max von Mises in the region of interest (hole + fillet)
    max_vm_global: float    # max over the whole body (affected by singularities)
    max_ux: float           # max |displacement| in x (mm)
    max_vm_root: float      # fillet / root region only
    max_vm_hole: float      # around the hole only
    loc_max: tuple          # (x, y, z) of the ROI maximum
    solve_time: float       # mesh import + assembly + solve (s)
    mesh: object = None
    uh: object = None
    vm: object = None


def solve(p, comm=MPI.COMM_WORLD):
    """Solve on the mesh currently held in gmsh.model (call after build_mesh)."""
    t_start = time.perf_counter()
    md = gmshio.model_to_mesh(gmsh.model, comm, 0)
    try:
        mesh, facet_tags = md.mesh, md.facet_tags
    except AttributeError:
        fields = [a for a in dir(md) if not a.startswith("_")]
        raise RuntimeError(f"Unexpected MeshData layout, fields are: {fields}")

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    V = fem.functionspace(mesh, ("Lagrange", 2, (3,)))

    # clamp: physical group 1 = bottom face of the base
    dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(1))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), dofs, V)

    mu = E / (2 * (1 + NU))
    lmbda = E * NU / ((1 + NU) * (1 - 2 * NU))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(3)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    one = fem.Constant(mesh, default_scalar_type(1.0))
    area = comm.allreduce(fem.assemble_scalar(fem.form(one * ds(2))), op=MPI.SUM)
    traction = fem.Constant(mesh, default_scalar_type((LOAD_N / area, 0.0, 0.0)))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(traction, v) * ds(2)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="bracket_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
    )
    uh = problem.solve()

    # von Mises, interpolated into DG1 (DG0 would underestimate peaks)
    s = sigma(uh) - (1.0 / 3.0) * ufl.tr(sigma(uh)) * ufl.Identity(3)
    vm_expr = ufl.sqrt(1.5 * ufl.inner(s, s))
    W = fem.functionspace(mesh, ("DG", 1))
    pts = W.element.interpolation_points
    pts = pts() if callable(pts) else pts      # method in older versions, property in newer
    vm = fem.Function(W)
    vm.interpolate(fem.Expression(vm_expr, pts))

    # region of interest: the lower leg incl. hole and fillet, away from the
    # clamped face and the loaded face (both have stress singularities)
    t, H, rh, rf = p["t"], p["H"], p["r_hole"], p["r_fillet"]
    z_hole = t + HOLE_FRAC * H
    xyz = W.tabulate_dof_coordinates()
    x, z = xyz[:, 0], xyz[:, 2]
    roi = (z >= t) & (z <= z_hole + rh + 3.0) & (x <= t + rf + 1.0)
    root = (z >= t) & (z <= t + rf + 2.0) & (x <= t + rf + 1.0)
    hole = (np.abs(z - z_hole) <= rh + 2.0) & (x <= t + 1.0)
    i_max = np.flatnonzero(roi)[np.argmax(vm.x.array[roi])]

    return FEAResult(
        ndof=V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        max_vm_roi=float(vm.x.array[roi].max()),
        max_vm_global=float(vm.x.array.max()),
        max_ux=float(np.abs(uh.x.array.reshape(-1, 3)[:, 0]).max()),
        max_vm_root=float(vm.x.array[root].max()),
        max_vm_hole=float(vm.x.array[hole].max()),
        loc_max=tuple(float(c) for c in xyz[i_max]),
        solve_time=time.perf_counter() - t_start,
        mesh=mesh, uh=uh, vm=vm,
    )
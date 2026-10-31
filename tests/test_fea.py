import gmsh
import pytest

from bracket.fea import E, LOAD_N, solve
from bracket.geometry import W, build_mesh

P = dict(t=6.0, H=45.0, r_hole=3.0, r_fillet=6.0)


@pytest.fixture(scope="module")
def result():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("t")
        build_mesh(P, 6.0, 1.5)
        return solve(P)
    finally:
        gmsh.finalize()


def test_tip_deflection_matches_beam_theory(result):
    I = W * P["t"] ** 3 / 12
    beam = LOAD_N * P["H"] ** 3 / (3 * E * I)
    assert 0.9 < result.max_ux / beam < 1.1


def test_peak_stress_close_to_nominal_bending_stress(result):
    nominal = 6 * LOAD_N * P["H"] / (W * P["t"] ** 2)
    assert 0.9 < result.max_vm_roi / nominal < 1.3


def test_global_max_is_not_below_roi_max(result):
    assert result.max_vm_global >= result.max_vm_roi
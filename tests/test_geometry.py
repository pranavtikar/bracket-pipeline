import gmsh
import pytest

from bracket.geometry import build_mesh, is_valid
from bracket.quality import quality_metrics

BASE = dict(t=6.0, H=45.0, r_hole=5.0, r_fillet=6.0)


@pytest.fixture
def clean_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def test_baseline_is_valid():
    assert is_valid(BASE)


def test_fillet_larger_than_half_the_leg_is_invalid():
    assert not is_valid({**BASE, "r_fillet": 30.0})


def test_hole_overlapping_fillet_is_invalid():
    assert not is_valid(dict(t=4.0, H=30.0, r_hole=8.0, r_fillet=10.0))


def test_invalid_geometry_raises(clean_gmsh):
    gmsh.model.add("t")
    with pytest.raises(ValueError):
        build_mesh({**BASE, "r_fillet": 30.0}, 8.0, 2.0)


def test_mesh_is_generated_with_sane_quality(clean_gmsh):
    gmsh.model.add("t")
    build_mesh(BASE, 8.0, 2.0)
    q = quality_metrics()
    assert q["n_tets"] > 500
    assert q["aspect_max"] >= 1.0
    assert 0.0 <= q["skew_max"] <= 1.0
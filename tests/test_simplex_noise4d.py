import pytest
from JaxOpenSimplexNoise.simplex_noise4d import Simplex2Jax4d

# JaxOpenSimplexNoise/test_simplex_noise4d.py
import jax
from JaxOpenSimplexNoise.types import jnpFloat32
import jax.numpy as jnp

@pytest.fixture(scope="module", autouse=True)
def setup():
    Simplex2Jax4d.initialize_grad4()

def test_initialize_grad4():
    assert Simplex2Jax4d.GRADIENTS_4D.shape[0] == Simplex2Jax4d.N_GRADS_4D * 4

def test_static_method():
    seed, _ = jax.random.split(jax.random.PRNGKey(seed=0))

    x = jnpFloat32([2_000, 2_000])
    y = jnpFloat32([-3_000, -3_000])
    z = jnpFloat32([5_000, 5_000])
    t = jnpFloat32([180, 180])
    noise = Simplex2Jax4d.noise4_Fallback(seed, x, y, z, t)
    assert isinstance(noise, jnp.ndarray)

def test_grad():
    seed = jnp.array(12345, dtype=jnp.int64)
    xsvp = jnp.array(2, dtype=jnp.int64)
    ysvp = jnp.array(3, dtype=jnp.int64)
    zsvp = jnp.array(4, dtype=jnp.int64)
    wsvp = jnp.array(5, dtype=jnp.int64)
    dx = jnp.array(0.5, dtype=jnp.float32)
    dy = jnp.array(0.5, dtype=jnp.float32)
    dz = jnp.array(0.5, dtype=jnp.float32)
    dw = jnp.array(0.5, dtype=jnp.float32)
    result = Simplex2Jax4d.grad(seed, xsvp, ysvp, zsvp, wsvp, dx, dy, dz, dw)
    assert isinstance(result, jnp.ndarray)

def test_noise4_Fallback():
    seed = jnp.array(12345, dtype=jnp.int64)
    x = jnp.array(0.5, dtype=jnp.float64)
    y = jnp.array(0.5, dtype=jnp.float64)
    z = jnp.array(0.5, dtype=jnp.float64)
    w = jnp.array(0.5, dtype=jnp.float64)
    result = Simplex2Jax4d.noise4_Fallback(seed, x, y, z, w)
    assert isinstance(result, jnp.ndarray)

def test_noise4_UnskewedBase():
    seed = jnp.array([12345, 67890], dtype=jnp.int64)
    xs = jnp.array([0.5, 0.6], dtype=jnp.float64)
    ys = jnp.array([0.5, 0.6], dtype=jnp.float64)
    zs = jnp.array([0.5, 0.6], dtype=jnp.float64)
    ws = jnp.array([0.5, 0.6], dtype=jnp.float64)
    result = Simplex2Jax4d.noise4_UnskewedBase(seed, xs, ys, zs, ws)
    assert isinstance(result, jnp.ndarray)

if __name__ == '__main__':
    pytest.main()
import jax.numpy as jnp

int_precision = jnp.int32
long_precision = jnp.int64
float_precision = jnp.float32
double_precision = jnp.float64

def jnpFloat32(x):
    return jnp.array(x, dtype=jnp.float32)
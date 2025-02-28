import math
import random
import time
# from ctypes import c_int64, c_int32
from typing import Union, Callable

import chex
import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import opensimplex
from jaxtyping import Float, Array, Int

from utils.type_alias_jnp import jnpFloat32

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit",False)

int_precision = jnp.int32
long_precision = jnp.int64
float_precision = jnp.float32
double_precision = jnp.float64
initialized = False


class Simplex2Jax4d():
    HASH_MULTIPLIER: int = jnp.array(0x53A3F72DEEC546F5, dtype=long_precision)
    N_GRADS_4D_EXPONENT: int = jnp.array(9, dtype=long_precision)
    N_GRADS_4D: int = jnp.array(1 << N_GRADS_4D_EXPONENT, dtype=long_precision)
    GRADIENTS_4D: jax.Array = jnp.zeros(N_GRADS_4D * 4, dtype=long_precision)
    SEED_OFFSET_4D: int = jnp.array(0xE83DC3E0DA7164D, dtype=long_precision)
    LATTICE_STEP_4D: float = jnp.array(0.2, dtype=double_precision)
    UNSKEW_4D: float = jnp.array(0.309016994374947, dtype=double_precision)
    PRIME_X: int = jnp.array(0x5205402B9270C86F, dtype=long_precision)
    PRIME_Y: int = jnp.array(0x598CD327003817B5, dtype=long_precision)
    PRIME_Z: int = jnp.array(0x5BCC226E9FA0BACB, dtype=long_precision)
    PRIME_W: int = jnp.array(0x56CC5227E58F554B, dtype=long_precision)
    RSQUARED_4D: float = jnp.array(0.6, dtype=double_precision)
    LATTICE_STEP_4D: float = jnp.array(0.2, dtype=double_precision)
    SKEW_4D: float_precision = jnp.array(-0.138196601125011, dtype=double_precision)
    NORMALIZER_4D: float = jnp.array(0.0220065933241897, dtype=double_precision)

    @classmethod
    def initialize_grad4(cls):
        grad4: jax.Array = np.array([
            -0.6740059517812944, -0.3239847771997537, -0.3239847771997537, 0.5794684678643381,
            -0.7504883828755602, -0.4004672082940195, 0.15296486218853164, 0.5029860367700724,
            -0.7504883828755602, 0.15296486218853164, -0.4004672082940195, 0.5029860367700724,
            -0.8828161875373585, 0.08164729285680945, 0.08164729285680945, 0.4553054119602712,
            -0.4553054119602712, -0.08164729285680945, -0.08164729285680945, 0.8828161875373585,
            -0.5029860367700724, -0.15296486218853164, 0.4004672082940195, 0.7504883828755602,
            -0.5029860367700724, 0.4004672082940195, -0.15296486218853164, 0.7504883828755602,
            -0.5794684678643381, 0.3239847771997537, 0.3239847771997537, 0.6740059517812944,
            -0.6740059517812944, -0.3239847771997537, 0.5794684678643381, -0.3239847771997537,
            -0.7504883828755602, -0.4004672082940195, 0.5029860367700724, 0.15296486218853164,
            -0.7504883828755602, 0.15296486218853164, 0.5029860367700724, -0.4004672082940195,
            -0.8828161875373585, 0.08164729285680945, 0.4553054119602712, 0.08164729285680945,
            -0.4553054119602712, -0.08164729285680945, 0.8828161875373585, -0.08164729285680945,
            -0.5029860367700724, -0.15296486218853164, 0.7504883828755602, 0.4004672082940195,
            -0.5029860367700724, 0.4004672082940195, 0.7504883828755602, -0.15296486218853164,
            -0.5794684678643381, 0.3239847771997537, 0.6740059517812944, 0.3239847771997537,
            -0.6740059517812944, 0.5794684678643381, -0.3239847771997537, -0.3239847771997537,
            -0.7504883828755602, 0.5029860367700724, -0.4004672082940195, 0.15296486218853164,
            -0.7504883828755602, 0.5029860367700724, 0.15296486218853164, -0.4004672082940195,
            -0.8828161875373585, 0.4553054119602712, 0.08164729285680945, 0.08164729285680945,
            -0.4553054119602712, 0.8828161875373585, -0.08164729285680945, -0.08164729285680945,
            -0.5029860367700724, 0.7504883828755602, -0.15296486218853164, 0.4004672082940195,
            -0.5029860367700724, 0.7504883828755602, 0.4004672082940195, -0.15296486218853164,
            -0.5794684678643381, 0.6740059517812944, 0.3239847771997537, 0.3239847771997537,
            0.5794684678643381, -0.6740059517812944, -0.3239847771997537, -0.3239847771997537,
            0.5029860367700724, -0.7504883828755602, -0.4004672082940195, 0.15296486218853164,
            0.5029860367700724, -0.7504883828755602, 0.15296486218853164, -0.4004672082940195,
            0.4553054119602712, -0.8828161875373585, 0.08164729285680945, 0.08164729285680945,
            0.8828161875373585, -0.4553054119602712, -0.08164729285680945, -0.08164729285680945,
            0.7504883828755602, -0.5029860367700724, -0.15296486218853164, 0.4004672082940195,
            0.7504883828755602, -0.5029860367700724, 0.4004672082940195, -0.15296486218853164,
            0.6740059517812944, -0.5794684678643381, 0.3239847771997537, 0.3239847771997537,
            -0.753341017856078, -0.37968289875261624, -0.37968289875261624, -0.37968289875261624,
            -0.7821684431180708, -0.4321472685365301, -0.4321472685365301, 0.12128480194602098,
            -0.7821684431180708, -0.4321472685365301, 0.12128480194602098, -0.4321472685365301,
            -0.7821684431180708, 0.12128480194602098, -0.4321472685365301, -0.4321472685365301,
            -0.8586508742123365, -0.508629699630796, 0.044802370851755174, 0.044802370851755174,
            -0.8586508742123365, 0.044802370851755174, -0.508629699630796, 0.044802370851755174,
            -0.8586508742123365, 0.044802370851755174, 0.044802370851755174, -0.508629699630796,
            -0.9982828964265062, -0.03381941603233842, -0.03381941603233842, -0.03381941603233842,
            -0.37968289875261624, -0.753341017856078, -0.37968289875261624, -0.37968289875261624,
            -0.4321472685365301, -0.7821684431180708, -0.4321472685365301, 0.12128480194602098,
            -0.4321472685365301, -0.7821684431180708, 0.12128480194602098, -0.4321472685365301,
            0.12128480194602098, -0.7821684431180708, -0.4321472685365301, -0.4321472685365301,
            -0.508629699630796, -0.8586508742123365, 0.044802370851755174, 0.044802370851755174,
            0.044802370851755174, -0.8586508742123365, -0.508629699630796, 0.044802370851755174,
            0.044802370851755174, -0.8586508742123365, 0.044802370851755174, -0.508629699630796,
            -0.03381941603233842, -0.9982828964265062, -0.03381941603233842, -0.03381941603233842,
            -0.37968289875261624, -0.37968289875261624, -0.753341017856078, -0.37968289875261624,
            -0.4321472685365301, -0.4321472685365301, -0.7821684431180708, 0.12128480194602098,
            -0.4321472685365301, 0.12128480194602098, -0.7821684431180708, -0.4321472685365301,
            0.12128480194602098, -0.4321472685365301, -0.7821684431180708, -0.4321472685365301,
            -0.508629699630796, 0.044802370851755174, -0.8586508742123365, 0.044802370851755174,
            0.044802370851755174, -0.508629699630796, -0.8586508742123365, 0.044802370851755174,
            0.044802370851755174, 0.044802370851755174, -0.8586508742123365, -0.508629699630796,
            -0.03381941603233842, -0.03381941603233842, -0.9982828964265062, -0.03381941603233842,
            -0.37968289875261624, -0.37968289875261624, -0.37968289875261624, -0.753341017856078,
            -0.4321472685365301, -0.4321472685365301, 0.12128480194602098, -0.7821684431180708,
            -0.4321472685365301, 0.12128480194602098, -0.4321472685365301, -0.7821684431180708,
            0.12128480194602098, -0.4321472685365301, -0.4321472685365301, -0.7821684431180708,
            -0.508629699630796, 0.044802370851755174, 0.044802370851755174, -0.8586508742123365,
            0.044802370851755174, -0.508629699630796, 0.044802370851755174, -0.8586508742123365,
            0.044802370851755174, 0.044802370851755174, -0.508629699630796, -0.8586508742123365,
            -0.03381941603233842, -0.03381941603233842, -0.03381941603233842, -0.9982828964265062,
            -0.3239847771997537, -0.6740059517812944, -0.3239847771997537, 0.5794684678643381,
            -0.4004672082940195, -0.7504883828755602, 0.15296486218853164, 0.5029860367700724,
            0.15296486218853164, -0.7504883828755602, -0.4004672082940195, 0.5029860367700724,
            0.08164729285680945, -0.8828161875373585, 0.08164729285680945, 0.4553054119602712,
            -0.08164729285680945, -0.4553054119602712, -0.08164729285680945, 0.8828161875373585,
            -0.15296486218853164, -0.5029860367700724, 0.4004672082940195, 0.7504883828755602,
            0.4004672082940195, -0.5029860367700724, -0.15296486218853164, 0.7504883828755602,
            0.3239847771997537, -0.5794684678643381, 0.3239847771997537, 0.6740059517812944,
            -0.3239847771997537, -0.3239847771997537, -0.6740059517812944, 0.5794684678643381,
            -0.4004672082940195, 0.15296486218853164, -0.7504883828755602, 0.5029860367700724,
            0.15296486218853164, -0.4004672082940195, -0.7504883828755602, 0.5029860367700724,
            0.08164729285680945, 0.08164729285680945, -0.8828161875373585, 0.4553054119602712,
            -0.08164729285680945, -0.08164729285680945, -0.4553054119602712, 0.8828161875373585,
            -0.15296486218853164, 0.4004672082940195, -0.5029860367700724, 0.7504883828755602,
            0.4004672082940195, -0.15296486218853164, -0.5029860367700724, 0.7504883828755602,
            0.3239847771997537, 0.3239847771997537, -0.5794684678643381, 0.6740059517812944,
            -0.3239847771997537, -0.6740059517812944, 0.5794684678643381, -0.3239847771997537,
            -0.4004672082940195, -0.7504883828755602, 0.5029860367700724, 0.15296486218853164,
            0.15296486218853164, -0.7504883828755602, 0.5029860367700724, -0.4004672082940195,
            0.08164729285680945, -0.8828161875373585, 0.4553054119602712, 0.08164729285680945,
            -0.08164729285680945, -0.4553054119602712, 0.8828161875373585, -0.08164729285680945,
            -0.15296486218853164, -0.5029860367700724, 0.7504883828755602, 0.4004672082940195,
            0.4004672082940195, -0.5029860367700724, 0.7504883828755602, -0.15296486218853164,
            0.3239847771997537, -0.5794684678643381, 0.6740059517812944, 0.3239847771997537,
            -0.3239847771997537, -0.3239847771997537, 0.5794684678643381, -0.6740059517812944,
            -0.4004672082940195, 0.15296486218853164, 0.5029860367700724, -0.7504883828755602,
            0.15296486218853164, -0.4004672082940195, 0.5029860367700724, -0.7504883828755602,
            0.08164729285680945, 0.08164729285680945, 0.4553054119602712, -0.8828161875373585,
            -0.08164729285680945, -0.08164729285680945, 0.8828161875373585, -0.4553054119602712,
            -0.15296486218853164, 0.4004672082940195, 0.7504883828755602, -0.5029860367700724,
            0.4004672082940195, -0.15296486218853164, 0.7504883828755602, -0.5029860367700724,
            0.3239847771997537, 0.3239847771997537, 0.6740059517812944, -0.5794684678643381,
            -0.3239847771997537, 0.5794684678643381, -0.6740059517812944, -0.3239847771997537,
            -0.4004672082940195, 0.5029860367700724, -0.7504883828755602, 0.15296486218853164,
            0.15296486218853164, 0.5029860367700724, -0.7504883828755602, -0.4004672082940195,
            0.08164729285680945, 0.4553054119602712, -0.8828161875373585, 0.08164729285680945,
            -0.08164729285680945, 0.8828161875373585, -0.4553054119602712, -0.08164729285680945,
            -0.15296486218853164, 0.7504883828755602, -0.5029860367700724, 0.4004672082940195,
            0.4004672082940195, 0.7504883828755602, -0.5029860367700724, -0.15296486218853164,
            0.3239847771997537, 0.6740059517812944, -0.5794684678643381, 0.3239847771997537,
            -0.3239847771997537, 0.5794684678643381, -0.3239847771997537, -0.6740059517812944,
            -0.4004672082940195, 0.5029860367700724, 0.15296486218853164, -0.7504883828755602,
            0.15296486218853164, 0.5029860367700724, -0.4004672082940195, -0.7504883828755602,
            0.08164729285680945, 0.4553054119602712, 0.08164729285680945, -0.8828161875373585,
            -0.08164729285680945, 0.8828161875373585, -0.08164729285680945, -0.4553054119602712,
            -0.15296486218853164, 0.7504883828755602, 0.4004672082940195, -0.5029860367700724,
            0.4004672082940195, 0.7504883828755602, -0.15296486218853164, -0.5029860367700724,
            0.3239847771997537, 0.6740059517812944, 0.3239847771997537, -0.5794684678643381,
            0.5794684678643381, -0.3239847771997537, -0.6740059517812944, -0.3239847771997537,
            0.5029860367700724, -0.4004672082940195, -0.7504883828755602, 0.15296486218853164,
            0.5029860367700724, 0.15296486218853164, -0.7504883828755602, -0.4004672082940195,
            0.4553054119602712, 0.08164729285680945, -0.8828161875373585, 0.08164729285680945,
            0.8828161875373585, -0.08164729285680945, -0.4553054119602712, -0.08164729285680945,
            0.7504883828755602, -0.15296486218853164, -0.5029860367700724, 0.4004672082940195,
            0.7504883828755602, 0.4004672082940195, -0.5029860367700724, -0.15296486218853164,
            0.6740059517812944, 0.3239847771997537, -0.5794684678643381, 0.3239847771997537,
            0.5794684678643381, -0.3239847771997537, -0.3239847771997537, -0.6740059517812944,
            0.5029860367700724, -0.4004672082940195, 0.15296486218853164, -0.7504883828755602,
            0.5029860367700724, 0.15296486218853164, -0.4004672082940195, -0.7504883828755602,
            0.4553054119602712, 0.08164729285680945, 0.08164729285680945, -0.8828161875373585,
            0.8828161875373585, -0.08164729285680945, -0.08164729285680945, -0.4553054119602712,
            0.7504883828755602, -0.15296486218853164, 0.4004672082940195, -0.5029860367700724,
            0.7504883828755602, 0.4004672082940195, -0.15296486218853164, -0.5029860367700724,
            0.6740059517812944, 0.3239847771997537, 0.3239847771997537, -0.5794684678643381,
            0.03381941603233842, 0.03381941603233842, 0.03381941603233842, 0.9982828964265062,
            -0.044802370851755174, -0.044802370851755174, 0.508629699630796, 0.8586508742123365,
            -0.044802370851755174, 0.508629699630796, -0.044802370851755174, 0.8586508742123365,
            -0.12128480194602098, 0.4321472685365301, 0.4321472685365301, 0.7821684431180708,
            0.508629699630796, -0.044802370851755174, -0.044802370851755174, 0.8586508742123365,
            0.4321472685365301, -0.12128480194602098, 0.4321472685365301, 0.7821684431180708,
            0.4321472685365301, 0.4321472685365301, -0.12128480194602098, 0.7821684431180708,
            0.37968289875261624, 0.37968289875261624, 0.37968289875261624, 0.753341017856078,
            0.03381941603233842, 0.03381941603233842, 0.9982828964265062, 0.03381941603233842,
            -0.044802370851755174, 0.044802370851755174, 0.8586508742123365, 0.508629699630796,
            -0.044802370851755174, 0.508629699630796, 0.8586508742123365, -0.044802370851755174,
            -0.12128480194602098, 0.4321472685365301, 0.7821684431180708, 0.4321472685365301,
            0.508629699630796, -0.044802370851755174, 0.8586508742123365, -0.044802370851755174,
            0.4321472685365301, -0.12128480194602098, 0.7821684431180708, 0.4321472685365301,
            0.4321472685365301, 0.4321472685365301, 0.7821684431180708, -0.12128480194602098,
            0.37968289875261624, 0.37968289875261624, 0.753341017856078, 0.37968289875261624,
            0.03381941603233842, 0.9982828964265062, 0.03381941603233842, 0.03381941603233842,
            -0.044802370851755174, 0.8586508742123365, -0.044802370851755174, 0.508629699630796,
            -0.044802370851755174, 0.8586508742123365, 0.508629699630796, -0.044802370851755174,
            -0.12128480194602098, 0.7821684431180708, 0.4321472685365301, 0.4321472685365301,
            0.508629699630796, 0.8586508742123365, -0.044802370851755174, -0.044802370851755174,
            0.4321472685365301, 0.7821684431180708, -0.12128480194602098, 0.4321472685365301,
            0.4321472685365301, 0.7821684431180708, 0.4321472685365301, -0.12128480194602098,
            0.37968289875261624, 0.753341017856078, 0.37968289875261624, 0.37968289875261624,
            0.9982828964265062, 0.03381941603233842, 0.03381941603233842, 0.03381941603233842,
            0.8586508742123365, -0.044802370851755174, -0.044802370851755174, 0.508629699630796,
            0.8586508742123365, -0.044802370851755174, 0.508629699630796, -0.044802370851755174,
            0.7821684431180708, -0.12128480194602098, 0.4321472685365301, 0.4321472685365301,
            0.8586508742123365, 0.508629699630796, -0.044802370851755174, -0.044802370851755174,
            0.7821684431180708, 0.4321472685365301, -0.12128480194602098, 0.4321472685365301,
            0.7821684431180708, 0.4321472685365301, 0.4321472685365301, -0.12128480194602098,
            0.753341017856078, 0.37968289875261624, 0.37968289875261624, 0.37968289875261624],
            dtype=float_precision) / cls.NORMALIZER_4D

        GRADIENTS_4D: jax.Array = jnp.zeros(cls.N_GRADS_4D * 4)
        j = 0
        for i in range(GRADIENTS_4D.shape[0]):
            if j == grad4.shape[0]:
                j = 0
            GRADIENTS_4D = GRADIENTS_4D.at[i].set(grad4[j])
            j += 1
        cls.GRADIENTS_4D = GRADIENTS_4D

    @staticmethod
    def grad(seed: Int[Array, ""],
             xsvp: Int[Array, ""], ysvp: Int[Array, ""], zsvp: Int[Array, ""], wsvp: Int[Array, ""],
             dx: Float[Array, ""], dy: Float[Array, ""], dz: Float[Array, ""], dw: Float[Array, ""]) -> Float[
        Array, ""]:
        hash: long_precision = jnp.bitwise_xor(jnp.bitwise_xor(seed, jnp.bitwise_xor(xsvp, ysvp)),
                                               (jnp.bitwise_xor(zsvp, wsvp))).astype(long_precision)
        hash *= Simplex2Jax4d.HASH_MULTIPLIER
        hash ^= hash >> (64 - Simplex2Jax4d.N_GRADS_4D_EXPONENT + 2)
        gi: int = hash.astype(int_precision) & ((Simplex2Jax4d.N_GRADS_4D - 1) << 2)
        return (Simplex2Jax4d.GRADIENTS_4D[gi | 0] * dx + Simplex2Jax4d.GRADIENTS_4D[gi | 1] * dy) + (
                Simplex2Jax4d.GRADIENTS_4D[gi | 2] * dz + Simplex2Jax4d.GRADIENTS_4D[gi | 3] * dw)

    @staticmethod
    def noise4_Fallback(seed: long_precision, x: double_precision, y: double_precision, z: double_precision,
                        w: double_precision):
        # Get points for A4 lattice
        s: double_precision = Simplex2Jax4d.SKEW_4D * (x + y + z + w)
        xs: double_precision = x + s;
        ys: double_precision = y + s;
        zs: double_precision = z + s;
        ws: double_precision = w + s
        return Simplex2Jax4d.noise4_UnskewedBase(seed.astype(long_precision), xs.astype(double_precision),
                                                 ys.astype(double_precision), zs.astype(double_precision),
                                                 ws.astype(double_precision))

    @staticmethod
    def next_point(i: int, ssi: float, xsvp: int, ysvp: int, zsvp: int, wsvp: int, xsi: float, ysi: float, zsi: float,
                   wsi: float, startingLattice: int, seed: int,
                   UNSKEW_4D: float, PRIME_X: int, PRIME_Y: int, PRIME_Z: int, PRIME_W: int, RSQUARED_4D: float,
                   LATTICE_STEP_4D: float, SEED_OFFSET_4D: int,
                   value: float, grad: Callable):
        # Next point is the closest vertex on the 4-simplex whose base vertex is the aforementioned vertex.
        score0 = 1.0 + ssi * (-1.0 / UNSKEW_4D)

        chained_if = jnp.ones(xsi.shape[0]).astype(bool)
        xsvp_next = jnp.where((xsi >= ysi) & (xsi >= zsi) & (xsi >= wsi) & (xsi >= score0) & chained_if, xsvp + PRIME_X,
                              xsvp)
        ssi = jnp.where((xsi >= ysi) & (xsi >= zsi) & (xsi >= wsi) & (xsi >= score0) & chained_if, ssi - UNSKEW_4D, ssi)
        xsi_next = jnp.where((xsi >= ysi) & (xsi >= zsi) & (xsi >= wsi) & (xsi >= score0) & chained_if, xsi - 1, xsi)
        chained_if = jnp.where((xsi >= ysi) & (xsi >= zsi) & (xsi >= wsi) & (xsi >= score0),
                               jnp.zeros(xsi.shape[0]).astype(bool), chained_if)

        ysvp_next = jnp.where((ysi > xsi) & (ysi >= zsi) & (ysi >= wsi) & (ysi >= score0) & chained_if, ysvp + PRIME_Y,
                              ysvp)
        ssi = jnp.where((ysi > xsi) & (ysi >= zsi) & (ysi >= wsi) & (ysi >= score0) & chained_if, ssi - UNSKEW_4D, ssi)
        ysi_next = jnp.where((ysi > xsi) & (ysi >= zsi) & (ysi >= wsi) & (ysi >= score0) & chained_if, ysi - 1, ysi)
        chained_if = jnp.where((ysi > xsi) & (ysi >= zsi) & (ysi >= wsi) & (ysi >= score0),
                               jnp.zeros(xsi.shape[0]).astype(bool), chained_if)

        zsvp_next = jnp.where((zsi > xsi) & (zsi > ysi) & (zsi >= wsi) & (zsi >= score0) & chained_if, zsvp + PRIME_Z,
                              zsvp)
        ssi = jnp.where((zsi > xsi) & (zsi > ysi) & (zsi >= wsi) & (zsi >= score0) & chained_if, ssi - UNSKEW_4D, ssi)
        zsi_next = jnp.where((zsi > xsi) & (zsi > ysi) & (zsi >= wsi) & (zsi >= score0) & chained_if, zsi - 1, zsi)
        chained_if = jnp.where((zsi > xsi) & (zsi > ysi) & (zsi >= wsi) & (zsi >= score0),
                               jnp.zeros(xsi.shape[0]).astype(bool), chained_if)

        wsvp_next = jnp.where((wsi > xsi) & (wsi > ysi) & (wsi > zsi) & (wsi >= score0) & chained_if, wsvp + PRIME_W,
                              wsvp)
        ssi = jnp.where((wsi > xsi) & (wsi > ysi) & (wsi > zsi) & (wsi >= score0) & chained_if, ssi - UNSKEW_4D, ssi)
        wsi_next = jnp.where((wsi > xsi) & (wsi > ysi) & (wsi > zsi) & (wsi >= score0) & chained_if, wsi - 1, wsi)

        # ssi_next_matrix = jnp.concatenate((ssi_next_1[:, None], ssi_next_2[:, None], ssi_next_3[:, None], ssi_next_4[:, None]), axis=1)

        xsvp = xsvp_next
        xsi = xsi_next
        ysvp = ysvp_next
        ysi = ysi_next
        zsvp = zsvp_next
        zsi = zsi_next
        wsvp = wsvp_next
        wsi = wsi_next

        # gradient contribution with falloff
        dx = xsi + ssi;
        dy = ysi + ssi;
        dz = zsi + ssi;
        dw = wsi + ssi
        a = (dx * dx + dy * dy) + (dz * dz + dw * dw)  # incorrect!!!!
        a_before = a
        a = jnp.where(a_before < RSQUARED_4D, (a - RSQUARED_4D) * (a - RSQUARED_4D), a)
        # grad_val = grad(seed, xsvp.astype(int), ysvp.astype(int), zsvp.astype(int), wsvp.astype(int), dx, dy, dz, dw)
        value = jnp.where(a_before < RSQUARED_4D,
                          value + a * a * grad(seed, xsvp.astype(long_precision), ysvp.astype(long_precision),
                                               zsvp.astype(long_precision), wsvp.astype(long_precision), dx, dy, dz,
                                               dw), value)

        # if i >= 4 don't progress through here.
        # update for next lattice copy shifted down by <-0.2, -0.2, -0.2, -0.2>
        # xsi = jnp.where(i < 4, xsi + LATTICE_STEP_4D, xsi);
        mask = i < 4
        updates = jnp.array([
            LATTICE_STEP_4D,
            LATTICE_STEP_4D,
            LATTICE_STEP_4D,
            LATTICE_STEP_4D,
            LATTICE_STEP_4D * 4 * UNSKEW_4D,
        ])
        current = jnp.array([xsi, ysi, zsi, wsi, ssi])
        xsi, ysi, zsi, wsi, ssi = jnp.where(mask, current + updates[:, None], current)

        mask2 = jnp.tile(jnp.array([i < 4]), startingLattice.shape[0]) & (i == startingLattice)
        updates2 = jnp.array([
            -PRIME_X,
            -PRIME_Y,
            -PRIME_Z,
            -PRIME_W,
        ])
        current2 = jnp.array([xsvp, ysvp, zsvp, wsvp])
        xsvp, ysvp, zsvp, wsvp = jnp.where(mask2, current2 + updates2[:, None], current2)

        seed = jnp.where(mask, seed - SEED_OFFSET_4D, seed)
        seed = jnp.where((i < 4) & (i == startingLattice), seed + SEED_OFFSET_4D * 5, seed)

        return (ssi, xsi, ysi, zsi, wsi,
                xsvp, ysvp, zsvp, wsvp,
                seed, value)

    # @chex.assert_max_traces(n=10)
    @staticmethod
    def noise4_UnskewedBase(seed: Union[int | jnp.ndarray], xs: Union[float | jnp.ndarray],
                            ys: Union[float | jnp.ndarray], zs: Union[float | jnp.ndarray],
                            ws: Union[float | jnp.ndarray]):
        # base points and offsets
        xsb = jnp.floor(xs).astype(int_precision);
        ysb = jnp.floor(ys).astype(int_precision);
        zsb = jnp.floor(zs).astype(int_precision);
        wsb = jnp.floor(ws).astype(int_precision)
        xsi = (xs - xsb).astype(float_precision);
        ysi = (ys - ysb).astype(float_precision);
        zsi = (zs - zsb).astype(float_precision);
        wsi = (ws - wsb).astype(float_precision)

        # determine which lattice has a contributing point its corresponding cell's base simplex.
        # only look at the spaces between the diagonal places
        siSum = (xsi + ysi) + (zsi + wsi)  # float
        startingLattice = (siSum * 1.25).astype(long_precision)

        # offset for seed based on first lattice copy
        seed += startingLattice * Simplex2Jax4d.SEED_OFFSET_4D

        # offset for lattrice point relative positions (skewed)
        startingLatticeOffset = (startingLattice).astype(float_precision) * -Simplex2Jax4d.LATTICE_STEP_4D
        xsi += startingLatticeOffset;
        ysi += startingLatticeOffset;
        zsi += startingLatticeOffset;
        wsi += startingLatticeOffset

        # prep for vertex contributions
        ssi = (siSum + startingLatticeOffset * 4) * Simplex2Jax4d.UNSKEW_4D

        # prime pre-multiplication for hash
        xsvp = (xsb * Simplex2Jax4d.PRIME_X).astype(long_precision);
        ysvp = (ysb * Simplex2Jax4d.PRIME_Y).astype(long_precision);
        zsvp = (zsb * Simplex2Jax4d.PRIME_Z).astype(long_precision);
        wsvp = (wsb * Simplex2Jax4d.PRIME_W).astype(long_precision)
        value = jnp.zeros(xs.shape[0]).astype(float_precision)

        ssi, xsi, ysi, zsi, wsi, xsvp, ysvp, zsvp, wsvp, seed, value = Simplex2Jax4d.next_point(
            0, ssi, xsvp, ysvp, zsvp, wsvp, xsi, ysi, zsi, wsi, startingLattice, seed,
            Simplex2Jax4d.UNSKEW_4D, Simplex2Jax4d.PRIME_X, Simplex2Jax4d.PRIME_Y, Simplex2Jax4d.PRIME_Z,
            Simplex2Jax4d.PRIME_W, Simplex2Jax4d.RSQUARED_4D, Simplex2Jax4d.LATTICE_STEP_4D,
            Simplex2Jax4d.SEED_OFFSET_4D,
            value, Simplex2Jax4d.grad
        )
        ssi, xsi, ysi, zsi, wsi, xsvp, ysvp, zsvp, wsvp, seed, value = Simplex2Jax4d.next_point(
            1, ssi, xsvp, ysvp, zsvp, wsvp, xsi, ysi, zsi, wsi, startingLattice, seed,
            Simplex2Jax4d.UNSKEW_4D, Simplex2Jax4d.PRIME_X, Simplex2Jax4d.PRIME_Y, Simplex2Jax4d.PRIME_Z,
            Simplex2Jax4d.PRIME_W, Simplex2Jax4d.RSQUARED_4D, Simplex2Jax4d.LATTICE_STEP_4D,
            Simplex2Jax4d.SEED_OFFSET_4D,
            value, Simplex2Jax4d.grad
        )
        ssi, xsi, ysi, zsi, wsi, xsvp, ysvp, zsvp, wsvp, seed, value = Simplex2Jax4d.next_point(
            2, ssi, xsvp, ysvp, zsvp, wsvp, xsi, ysi, zsi, wsi, startingLattice, seed,
            Simplex2Jax4d.UNSKEW_4D, Simplex2Jax4d.PRIME_X, Simplex2Jax4d.PRIME_Y, Simplex2Jax4d.PRIME_Z,
            Simplex2Jax4d.PRIME_W, Simplex2Jax4d.RSQUARED_4D, Simplex2Jax4d.LATTICE_STEP_4D,
            Simplex2Jax4d.SEED_OFFSET_4D,
            value, Simplex2Jax4d.grad
        )
        ssi, xsi, ysi, zsi, wsi, xsvp, ysvp, zsvp, wsvp, seed, value = Simplex2Jax4d.next_point(
            3, ssi, xsvp, ysvp, zsvp, wsvp, xsi, ysi, zsi, wsi, startingLattice, seed,
            Simplex2Jax4d.UNSKEW_4D, Simplex2Jax4d.PRIME_X, Simplex2Jax4d.PRIME_Y, Simplex2Jax4d.PRIME_Z,
            Simplex2Jax4d.PRIME_W, Simplex2Jax4d.RSQUARED_4D, Simplex2Jax4d.LATTICE_STEP_4D,
            Simplex2Jax4d.SEED_OFFSET_4D,
            value, Simplex2Jax4d.grad
        )
        ssi, xsi, ysi, zsi, wsi, xsvp, ysvp, zsvp, wsvp, seed, value = Simplex2Jax4d.next_point(
            4, ssi, xsvp, ysvp, zsvp, wsvp, xsi, ysi, zsi, wsi, startingLattice, seed,
            Simplex2Jax4d.UNSKEW_4D, Simplex2Jax4d.PRIME_X, Simplex2Jax4d.PRIME_Y, Simplex2Jax4d.PRIME_Z,
            Simplex2Jax4d.PRIME_W, Simplex2Jax4d.RSQUARED_4D, Simplex2Jax4d.LATTICE_STEP_4D,
            Simplex2Jax4d.SEED_OFFSET_4D,
            value, Simplex2Jax4d.grad
        )
        return value


Simplex2Jax4d.initialize_grad4()

'''

    ====================================== Experiments ======================================

'''

if __name__ == "__main__":
    def get_value(seed: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, w: jnp.ndarray):
        return simplex2jax4d.noise4_Fallback(seed=seed, x=x, y=y, z=z, w=w)


    run: str = 'eval_static_method'
    simplex2jax4d = Simplex2Jax4d()
    simplex_generator = opensimplex.OpenSimplex(seed=3)

    if run == "eval_static_method":
        seed, _ = jax.random.split(jax.random.PRNGKey(seed=0))

        x = jnpFloat32([2_000, 2_000])
        y = jnpFloat32([-3_000, -3_000])
        z = jnpFloat32([5_000, 5_000])
        t = jnpFloat32([180, 180])
        noise = Simplex2Jax4d.noise4_Fallback(seed, x, y, z, t)
        print(noise)
        exit(1)

    if run == 'visualise':
        scale = 10
        height, width = 300, 300
        height_px, width_px = jnp.arange(height) / scale, jnp.arange(width) / scale
        X, Y = jnp.meshgrid(height_px, width_px)
        x, y = X.flatten(), Y.flatten()
        num_frames = 600

        for frame_number in range(num_frames):
            val = simplex2jax4d.noise4_Fallback(seed=jnp.ones(len(x)).astype(long_precision) * 3,
                                                x=x.astype(double_precision) / scale,
                                                y=y.astype(double_precision) / scale,
                                                z=jnp.ones(len(x)).astype(double_precision),
                                                w=jnp.ones(len(x)).astype(double_precision) + frame_number / num_frames)

            val = (val + 1) / 2
            val = (val > 0.5).astype(float)
            image = val.reshape(X.shape)

            # gamma = 0.5
            # corrected = jnp.power(val, gamma)
            # val = jnp.clip(val , a_min=0.0, a_max=1.0)

            # p1, p99 = jnp.percentile(val, (30, 70))
            # data = jnp.clip((val - p1) / (p99 - p1), 0, 1)

            # val = (val > 0.5).astype(float)

            # val = jnp.clip(val, a_min=-0.5, a_max=0.5)
            # print(jnp.max(val), jnp.min(val))
            # val_normalized = ((val - (-0.5)) / ((0.5) - (-0.5))).astype(float_precision)
            # print(jnp.max(val_normalized), jnp.min(val_normalized))
            image = val.reshape(X.shape)
            # image = cv2.resize(image, (300, 300))
            cv2.imshow('Dynamic Grayscale Image with Simplex Noise', image)
            if cv2.waitKey(25) == ord('q'):  # Wait for 100 ms or until 'q' is pressed
                break
        cv2.destroyAllWindows()
    elif run == 'test':
        val = simplex2jax4d.noise4_UnskewedBase(seed=jnp.ones(10, dtype=long_precision) * 3,
                                                xs=jnp.arange(10, dtype=double_precision) / 10,
                                                ys=jnp.arange(10, dtype=double_precision) / 10,
                                                zs=jnp.arange(10, dtype=double_precision) / 10,
                                                ws=jnp.arange(10, dtype=double_precision) / 10)
        print(val)
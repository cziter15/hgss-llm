"""Memory-efficient real-quaternion recurrent kernels for HGSS4.

The state is represented by four FP32 buffers (w, x, y, z).  One Triton
program owns one [batch, head, value, key] lane and scans the time dimension.
The legacy scan API is retained, but its backward maps one program to
[batch, head, key] and reduces value lanes locally instead of using atomics.

``triton_factorized_quaternion_read`` is the FlashHGSS-style API.  It builds
the low-rank injection on demand, contracts the recurrent state with the query
inside the forward kernel, stores only sparse boundary states, and replays one
chunk at a time during backward.  The full [B,T,H,V,K,4] history never exists.
"""

import math
import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


@triton.jit
def _scan_fwd(
    aw, ax, ay, az, bw, bx, by, bz,
    ow, ox, oy, oz,
    T: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
):
    lane = tl.program_id(0)
    k = lane % K
    q = lane // K
    v = q % V
    q = q // V
    h = q % H
    batch = q // H

    sw = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    for t in tl.range(0, T):
        ao = ((batch * T + t) * H + h) * K + k
        bo = (((batch * T + t) * H + h) * V + v) * K + k
        a0 = tl.load(aw + ao); a1 = tl.load(ax + ao)
        a2 = tl.load(ay + ao); a3 = tl.load(az + ao)
        n0 = a0*sw - a1*sx - a2*sy - a3*sz + tl.load(bw + bo)
        n1 = a0*sx + a1*sw + a2*sz - a3*sy + tl.load(bx + bo)
        n2 = a0*sy - a1*sz + a2*sw + a3*sx + tl.load(by + bo)
        n3 = a0*sz + a1*sy - a2*sx + a3*sw + tl.load(bz + bo)
        sw, sx, sy, sz = n0, n1, n2, n3
        tl.store(ow + bo, sw); tl.store(ox + bo, sx)
        tl.store(oy + bo, sy); tl.store(oz + bo, sz)


@triton.jit
def _scan_bwd(
    aw, ax, ay, az, ow, ox, oy, oz,
    gw, gx, gy, gz,
    daw, dax, day, daz, dbw, dbx, dby, dbz,
    T: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    lane = tl.program_id(0)
    k = lane % K
    q = lane // K
    h = q % H
    batch = q // H
    v = tl.arange(0, BLOCK_V)
    v_mask = v < V

    cw = tl.zeros((BLOCK_V,), tl.float32)
    cx = tl.zeros((BLOCK_V,), tl.float32)
    cy = tl.zeros((BLOCK_V,), tl.float32)
    cz = tl.zeros((BLOCK_V,), tl.float32)
    for rev in tl.range(0, T):
        t = T - 1 - rev
        ao = ((batch * T + t) * H + h) * K + k
        bo = (((batch * T + t) * H + h) * V + v) * K + k
        lw = tl.load(gw + bo, mask=v_mask, other=0.0) + cw
        lx = tl.load(gx + bo, mask=v_mask, other=0.0) + cx
        ly = tl.load(gy + bo, mask=v_mask, other=0.0) + cy
        lz = tl.load(gz + bo, mask=v_mask, other=0.0) + cz
        tl.store(dbw + bo, lw, mask=v_mask); tl.store(dbx + bo, lx, mask=v_mask)
        tl.store(dby + bo, ly, mask=v_mask); tl.store(dbz + bo, lz, mask=v_mask)

        po = (((batch * T + (t-1)) * H + h) * V + v) * K + k
        has_prev = (t > 0) & v_mask
        sw = tl.load(ow + po, mask=has_prev, other=0.0)
        sx = tl.load(ox + po, mask=has_prev, other=0.0)
        sy = tl.load(oy + po, mask=has_prev, other=0.0)
        sz = tl.load(oz + po, mask=has_prev, other=0.0)

        ga0 = lw*sw + lx*sx + ly*sy + lz*sz
        ga1 = -lw*sx + lx*sw - ly*sz + lz*sy
        ga2 = -lw*sy + lx*sz + ly*sw - lz*sx
        ga3 = -lw*sz - lx*sy + ly*sx + lz*sw
        tl.store(daw + ao, tl.sum(ga0, axis=0))
        tl.store(dax + ao, tl.sum(ga1, axis=0))
        tl.store(day + ao, tl.sum(ga2, axis=0))
        tl.store(daz + ao, tl.sum(ga3, axis=0))

        a0 = tl.load(aw + ao); a1 = tl.load(ax + ao)
        a2 = tl.load(ay + ao); a3 = tl.load(az + ao)
        cw = lw*a0 + lx*a1 + ly*a2 + lz*a3
        cx = -lw*a1 + lx*a0 + ly*a3 - lz*a2
        cy = -lw*a2 - lx*a3 + ly*a0 + lz*a1
        cz = -lw*a3 + lx*a2 - ly*a1 + lz*a0


@triton.jit
def _factorized_read_fwd(
    aw, ax, ay, az,
    qw, qx, qy, qz,
    kw, kx, ky, kz,
    vw, vx, vy, vz,
    inject, gate,
    cpw, cpx, cpy, cpz,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_READ: tl.constexpr,
):
    """Fused factorized recurrence, query contraction, gate, and checkpoints."""
    lane = tl.program_id(0)
    value_index = lane % V
    lane = lane // V
    head = lane % H
    batch = lane // H

    key_index = tl.arange(0, BLOCK_K)
    key_mask = key_index < K
    sw = tl.zeros((BLOCK_K,), tl.float32)
    sx = tl.zeros((BLOCK_K,), tl.float32)
    sy = tl.zeros((BLOCK_K,), tl.float32)
    sz = tl.zeros((BLOCK_K,), tl.float32)

    checkpoint_offset = (
        (((batch * (N_CHUNKS + 1)) * H + head) * V + value_index) * K
        + key_index
    )
    tl.store(cpw + checkpoint_offset, sw, mask=key_mask)
    tl.store(cpx + checkpoint_offset, sx, mask=key_mask)
    tl.store(cpy + checkpoint_offset, sy, mask=key_mask)
    tl.store(cpz + checkpoint_offset, sz, mask=key_mask)

    for chunk in tl.range(0, N_CHUNKS):
        for offset in tl.range(0, CHUNK_SIZE):
            time_index = chunk * CHUNK_SIZE + offset
            valid_time = time_index < T
            time_key_mask = key_mask & valid_time

            key_offset = ((batch * T + time_index) * H + head) * K + key_index
            value_offset = (((batch * T + time_index) * H + head) * V + value_index)

            a0 = tl.load(aw + key_offset, mask=time_key_mask, other=0.0)
            a1 = tl.load(ax + key_offset, mask=time_key_mask, other=0.0)
            a2 = tl.load(ay + key_offset, mask=time_key_mask, other=0.0)
            a3 = tl.load(az + key_offset, mask=time_key_mask, other=0.0)

            q0 = tl.load(qw + key_offset, mask=time_key_mask, other=0.0)
            q1 = tl.load(qx + key_offset, mask=time_key_mask, other=0.0)
            q2 = tl.load(qy + key_offset, mask=time_key_mask, other=0.0)
            q3 = tl.load(qz + key_offset, mask=time_key_mask, other=0.0)

            k0 = tl.load(kw + key_offset, mask=time_key_mask, other=0.0)
            k1 = tl.load(kx + key_offset, mask=time_key_mask, other=0.0)
            k2 = tl.load(ky + key_offset, mask=time_key_mask, other=0.0)
            k3 = tl.load(kz + key_offset, mask=time_key_mask, other=0.0)
            scale = tl.load(inject + key_offset, mask=time_key_mask, other=0.0)

            value0 = tl.load(vw + value_offset, mask=valid_time, other=0.0)
            value1 = tl.load(vx + value_offset, mask=valid_time, other=0.0)
            value2 = tl.load(vy + value_offset, mask=valid_time, other=0.0)
            value3 = tl.load(vz + value_offset, mask=valid_time, other=0.0)

            # conj(k) * inject, followed by v * (...), generated on demand.
            c0 = k0 * scale
            c1 = -k1 * scale
            c2 = -k2 * scale
            c3 = -k3 * scale
            b0 = value0*c0 - value1*c1 - value2*c2 - value3*c3
            b1 = value0*c1 + value1*c0 + value2*c3 - value3*c2
            b2 = value0*c2 - value1*c3 + value2*c0 + value3*c1
            b3 = value0*c3 + value1*c2 - value2*c1 + value3*c0

            n0 = a0*sw - a1*sx - a2*sy - a3*sz + b0
            n1 = a0*sx + a1*sw + a2*sz - a3*sy + b1
            n2 = a0*sy - a1*sz + a2*sw + a3*sx + b2
            n3 = a0*sz + a1*sy - a2*sx + a3*sw + b3
            sw = tl.where(valid_time, n0, sw)
            sx = tl.where(valid_time, n1, sx)
            sy = tl.where(valid_time, n2, sy)
            sz = tl.where(valid_time, n3, sz)

            r0 = tl.sum(sw*q0 - sx*q1 - sy*q2 - sz*q3, axis=0)
            r1 = tl.sum(sw*q1 + sx*q0 + sy*q3 - sz*q2, axis=0)
            r2 = tl.sum(sw*q2 - sx*q3 + sy*q0 + sz*q1, axis=0)
            r3 = tl.sum(sw*q3 + sx*q2 - sy*q1 + sz*q0, axis=0)
            gate_value = tl.load(gate + value_offset, mask=valid_time, other=0.0)
            if FULL_READ:
                output_offset = value_offset * 4
                tl.store(output + output_offset, r0 * gate_value, mask=valid_time)
                tl.store(output + output_offset + 1, r1 * gate_value, mask=valid_time)
                tl.store(output + output_offset + 2, r2 * gate_value, mask=valid_time)
                tl.store(output + output_offset + 3, r3 * gate_value, mask=valid_time)
            else:
                tl.store(output + value_offset, r0 * gate_value, mask=valid_time)

        checkpoint_offset = (
            ((((batch * (N_CHUNKS + 1) + chunk + 1) * H + head) * V + value_index) * K)
            + key_index
        )
        tl.store(cpw + checkpoint_offset, sw, mask=key_mask)
        tl.store(cpx + checkpoint_offset, sx, mask=key_mask)
        tl.store(cpy + checkpoint_offset, sy, mask=key_mask)
        tl.store(cpz + checkpoint_offset, sz, mask=key_mask)


@triton.jit
def _recompute_chunk_states(
    aw, ax, ay, az,
    kw, kx, ky, kz,
    vw, vx, vy, vz,
    inject,
    cpw, cpx, cpy, cpz,
    tw, tx, ty, tz,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_INDEX,
    START,
    C_LEN: tl.constexpr,
    TEMP_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reconstruct one chunk's state history into a reusable temporary buffer."""
    lane = tl.program_id(0)
    value_index = lane % V
    lane = lane // V
    head = lane % H
    batch = lane // H
    key_index = tl.arange(0, BLOCK_K)
    key_mask = key_index < K

    checkpoint_offset = (
        ((((batch * (N_CHUNKS + 1) + CHUNK_INDEX) * H + head) * V + value_index) * K)
        + key_index
    )
    sw = tl.load(cpw + checkpoint_offset, mask=key_mask, other=0.0)
    sx = tl.load(cpx + checkpoint_offset, mask=key_mask, other=0.0)
    sy = tl.load(cpy + checkpoint_offset, mask=key_mask, other=0.0)
    sz = tl.load(cpz + checkpoint_offset, mask=key_mask, other=0.0)

    for local_time in tl.range(0, C_LEN):
        time_index = START + local_time
        key_offset = ((batch * T + time_index) * H + head) * K + key_index
        value_offset = (((batch * T + time_index) * H + head) * V + value_index)
        a0 = tl.load(aw + key_offset, mask=key_mask, other=0.0)
        a1 = tl.load(ax + key_offset, mask=key_mask, other=0.0)
        a2 = tl.load(ay + key_offset, mask=key_mask, other=0.0)
        a3 = tl.load(az + key_offset, mask=key_mask, other=0.0)
        k0 = tl.load(kw + key_offset, mask=key_mask, other=0.0)
        k1 = tl.load(kx + key_offset, mask=key_mask, other=0.0)
        k2 = tl.load(ky + key_offset, mask=key_mask, other=0.0)
        k3 = tl.load(kz + key_offset, mask=key_mask, other=0.0)
        scale = tl.load(inject + key_offset, mask=key_mask, other=0.0)
        value0 = tl.load(vw + value_offset)
        value1 = tl.load(vx + value_offset)
        value2 = tl.load(vy + value_offset)
        value3 = tl.load(vz + value_offset)

        c0 = k0 * scale
        c1 = -k1 * scale
        c2 = -k2 * scale
        c3 = -k3 * scale
        b0 = value0*c0 - value1*c1 - value2*c2 - value3*c3
        b1 = value0*c1 + value1*c0 + value2*c3 - value3*c2
        b2 = value0*c2 - value1*c3 + value2*c0 + value3*c1
        b3 = value0*c3 + value1*c2 - value2*c1 + value3*c0
        n0 = a0*sw - a1*sx - a2*sy - a3*sz + b0
        n1 = a0*sx + a1*sw + a2*sz - a3*sy + b1
        n2 = a0*sy - a1*sz + a2*sw + a3*sx + b2
        n3 = a0*sz + a1*sy - a2*sx + a3*sw + b3
        sw, sx, sy, sz = n0, n1, n2, n3

        temp_offset = (
            ((((batch * TEMP_C + local_time) * H + head) * V + value_index) * K)
            + key_index
        )
        tl.store(tw + temp_offset, sw, mask=key_mask)
        tl.store(tx + temp_offset, sx, mask=key_mask)
        tl.store(ty + temp_offset, sy, mask=key_mask)
        tl.store(tz + temp_offset, sz, mask=key_mask)


@triton.jit
def _factorized_chunk_bwd(
    aw, ax, ay, az,
    qw, qx, qy, qz,
    kw, kx, ky, kz,
    vw, vx, vy, vz,
    inject, gate,
    cpw, cpx, cpy, cpz,
    tw, tx, ty, tz,
    grad_output,
    adjw, adjx, adjy, adjz,
    daw, dax, day, daz,
    dqw, dqx, dqy, dqz,
    dkw, dkx, dky, dkz,
    dvw, dvx, dvy, dvz,
    dinject, dgate,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_INDEX,
    START,
    C_LEN: tl.constexpr,
    TEMP_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FULL_READ: tl.constexpr,
):
    """Reverse one chunk; V is reduced locally and the state adjoint is carried."""
    lane = tl.program_id(0)
    key_index = lane % K
    lane = lane // K
    head = lane % H
    batch = lane // H
    value_index = tl.arange(0, BLOCK_V)
    value_mask = value_index < V

    state_offset = (((batch * H + head) * V + value_index) * K + key_index)
    carry0 = tl.load(adjw + state_offset, mask=value_mask, other=0.0)
    carry1 = tl.load(adjx + state_offset, mask=value_mask, other=0.0)
    carry2 = tl.load(adjy + state_offset, mask=value_mask, other=0.0)
    carry3 = tl.load(adjz + state_offset, mask=value_mask, other=0.0)

    for reverse_time in tl.range(0, C_LEN):
        local_time = C_LEN - 1 - reverse_time
        time_index = START + local_time
        key_offset = ((batch * T + time_index) * H + head) * K + key_index
        value_offset = (((batch * T + time_index) * H + head) * V + value_index)
        temp_offset = (
            ((((batch * TEMP_C + local_time) * H + head) * V + value_index) * K)
            + key_index
        )
        state0 = tl.load(tw + temp_offset, mask=value_mask, other=0.0)
        state1 = tl.load(tx + temp_offset, mask=value_mask, other=0.0)
        state2 = tl.load(ty + temp_offset, mask=value_mask, other=0.0)
        state3 = tl.load(tz + temp_offset, mask=value_mask, other=0.0)

        q0 = tl.load(qw + key_offset); q1 = tl.load(qx + key_offset)
        q2 = tl.load(qy + key_offset); q3 = tl.load(qz + key_offset)
        gate_value = tl.load(gate + value_offset, mask=value_mask, other=0.0)
        if FULL_READ:
            output_offset = value_offset * 4
            go0 = tl.load(grad_output + output_offset, mask=value_mask, other=0.0)
            go1 = tl.load(grad_output + output_offset + 1, mask=value_mask, other=0.0)
            go2 = tl.load(grad_output + output_offset + 2, mask=value_mask, other=0.0)
            go3 = tl.load(grad_output + output_offset + 3, mask=value_mask, other=0.0)
        else:
            go0 = tl.load(grad_output + value_offset, mask=value_mask, other=0.0)
            go1 = tl.zeros((BLOCK_V,), tl.float32)
            go2 = tl.zeros((BLOCK_V,), tl.float32)
            go3 = tl.zeros((BLOCK_V,), tl.float32)
        gr0 = go0 * gate_value
        gr1 = go1 * gate_value
        gr2 = go2 * gate_value
        gr3 = go3 * gate_value

        read0 = state0*q0 - state1*q1 - state2*q2 - state3*q3
        read1 = state0*q1 + state1*q0 + state2*q3 - state3*q2
        read2 = state0*q2 - state1*q3 + state2*q0 + state3*q1
        read3 = state0*q3 + state1*q2 - state2*q1 + state3*q0
        gate_partial = go0*read0 + go1*read1 + go2*read2 + go3*read3
        tl.atomic_add(dgate + value_offset, gate_partial, mask=value_mask)

        # Adjoint of state * query with respect to state and query.
        ds0 = gr0*q0 + gr1*q1 + gr2*q2 + gr3*q3
        ds1 = -gr0*q1 + gr1*q0 - gr2*q3 + gr3*q2
        ds2 = -gr0*q2 + gr1*q3 + gr2*q0 - gr3*q1
        ds3 = -gr0*q3 - gr1*q2 + gr2*q1 + gr3*q0
        dq0 = gr0*state0 + gr1*state1 + gr2*state2 + gr3*state3
        dq1 = -gr0*state1 + gr1*state0 + gr2*state3 - gr3*state2
        dq2 = -gr0*state2 - gr1*state3 + gr2*state0 + gr3*state1
        dq3 = -gr0*state3 + gr1*state2 - gr2*state1 + gr3*state0
        tl.store(dqw + key_offset, tl.sum(dq0, axis=0))
        tl.store(dqx + key_offset, tl.sum(dq1, axis=0))
        tl.store(dqy + key_offset, tl.sum(dq2, axis=0))
        tl.store(dqz + key_offset, tl.sum(dq3, axis=0))

        lam0 = carry0 + ds0; lam1 = carry1 + ds1
        lam2 = carry2 + ds2; lam3 = carry3 + ds3
        if local_time == 0:
            previous_offset = (
                ((((batch * (N_CHUNKS + 1) + CHUNK_INDEX) * H + head) * V + value_index) * K)
                + key_index
            )
            prev0 = tl.load(cpw + previous_offset, mask=value_mask, other=0.0)
            prev1 = tl.load(cpx + previous_offset, mask=value_mask, other=0.0)
            prev2 = tl.load(cpy + previous_offset, mask=value_mask, other=0.0)
            prev3 = tl.load(cpz + previous_offset, mask=value_mask, other=0.0)
        else:
            previous_offset = (
                ((((batch * TEMP_C + local_time - 1) * H + head) * V + value_index) * K)
                + key_index
            )
            prev0 = tl.load(tw + previous_offset, mask=value_mask, other=0.0)
            prev1 = tl.load(tx + previous_offset, mask=value_mask, other=0.0)
            prev2 = tl.load(ty + previous_offset, mask=value_mask, other=0.0)
            prev3 = tl.load(tz + previous_offset, mask=value_mask, other=0.0)

        da0 = lam0*prev0 + lam1*prev1 + lam2*prev2 + lam3*prev3
        da1 = -lam0*prev1 + lam1*prev0 - lam2*prev3 + lam3*prev2
        da2 = -lam0*prev2 + lam1*prev3 + lam2*prev0 - lam3*prev1
        da3 = -lam0*prev3 - lam1*prev2 + lam2*prev1 + lam3*prev0
        tl.store(daw + key_offset, tl.sum(da0, axis=0))
        tl.store(dax + key_offset, tl.sum(da1, axis=0))
        tl.store(day + key_offset, tl.sum(da2, axis=0))
        tl.store(daz + key_offset, tl.sum(da3, axis=0))

        key0 = tl.load(kw + key_offset); key1 = tl.load(kx + key_offset)
        key2 = tl.load(ky + key_offset); key3 = tl.load(kz + key_offset)
        scale = tl.load(inject + key_offset)
        c0 = key0*scale; c1 = -key1*scale
        c2 = -key2*scale; c3 = -key3*scale
        value0 = tl.load(vw + value_offset, mask=value_mask, other=0.0)
        value1 = tl.load(vx + value_offset, mask=value_mask, other=0.0)
        value2 = tl.load(vy + value_offset, mask=value_mask, other=0.0)
        value3 = tl.load(vz + value_offset, mask=value_mask, other=0.0)

        dv0 = lam0*c0 + lam1*c1 + lam2*c2 + lam3*c3
        dv1 = -lam0*c1 + lam1*c0 - lam2*c3 + lam3*c2
        dv2 = -lam0*c2 + lam1*c3 + lam2*c0 - lam3*c1
        dv3 = -lam0*c3 - lam1*c2 + lam2*c1 + lam3*c0
        tl.atomic_add(dvw + value_offset, dv0, mask=value_mask)
        tl.atomic_add(dvx + value_offset, dv1, mask=value_mask)
        tl.atomic_add(dvy + value_offset, dv2, mask=value_mask)
        tl.atomic_add(dvz + value_offset, dv3, mask=value_mask)

        dc0 = lam0*value0 + lam1*value1 + lam2*value2 + lam3*value3
        dc1 = -lam0*value1 + lam1*value0 + lam2*value3 - lam3*value2
        dc2 = -lam0*value2 - lam1*value3 + lam2*value0 + lam3*value1
        dc3 = -lam0*value3 + lam1*value2 - lam2*value1 + lam3*value0
        dc0 = tl.sum(dc0, axis=0); dc1 = tl.sum(dc1, axis=0)
        dc2 = tl.sum(dc2, axis=0); dc3 = tl.sum(dc3, axis=0)
        tl.store(dkw + key_offset, dc0*scale)
        tl.store(dkx + key_offset, -dc1*scale)
        tl.store(dky + key_offset, -dc2*scale)
        tl.store(dkz + key_offset, -dc3*scale)
        tl.store(dinject + key_offset, dc0*key0 - dc1*key1 - dc2*key2 - dc3*key3)

        a0 = tl.load(aw + key_offset); a1 = tl.load(ax + key_offset)
        a2 = tl.load(ay + key_offset); a3 = tl.load(az + key_offset)
        carry0 = lam0*a0 + lam1*a1 + lam2*a2 + lam3*a3
        carry1 = -lam0*a1 + lam1*a0 + lam2*a3 - lam3*a2
        carry2 = -lam0*a2 - lam1*a3 + lam2*a0 + lam3*a1
        carry3 = -lam0*a3 + lam1*a2 - lam2*a1 + lam3*a0

    tl.store(adjw + state_offset, carry0, mask=value_mask)
    tl.store(adjx + state_offset, carry1, mask=value_mask)
    tl.store(adjy + state_offset, carry2, mask=value_mask)
    tl.store(adjz + state_offset, carry3, mask=value_mask)


def _real_qmul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    )


def _torch_factorized_chunk(
    aw, ax, ay, az,
    qw, qx, qy, qz,
    kw, kx, ky, kz,
    vw, vx, vy, vz,
    inject, gate,
    statew, statex, statey, statez,
    full_read,
):
    """Autograd reference used to replay exactly one chunk in backward."""
    state = (statew, statex, statey, statez)
    outputs = []
    for time_index in range(aw.shape[1]):
        transition = tuple(
            component[:,time_index].unsqueeze(-2)
            for component in (aw, ax, ay, az)
        )
        query = (qw[:,time_index], qx[:,time_index], qy[:,time_index], qz[:,time_index])
        key_conjugate = (
            kw[:,time_index] * inject[:,time_index],
            -kx[:,time_index] * inject[:,time_index],
            -ky[:,time_index] * inject[:,time_index],
            -kz[:,time_index] * inject[:,time_index],
        )
        value = (
            vw[:,time_index].unsqueeze(-1), vx[:,time_index].unsqueeze(-1),
            vy[:,time_index].unsqueeze(-1), vz[:,time_index].unsqueeze(-1),
        )
        key_conjugate = tuple(component.unsqueeze(-2) for component in key_conjugate)
        injection_value = _real_qmul(value, key_conjugate)
        advanced = _real_qmul(transition, state)
        state = tuple(left + right for left, right in zip(advanced, injection_value))

        query = tuple(component.unsqueeze(-2) for component in query)
        read = _real_qmul(state, query)
        read = tuple(component.sum(-1) for component in read)
        gate_value = gate[:,time_index]
        if full_read:
            outputs.append(torch.stack(read, dim=-1) * gate_value.unsqueeze(-1))
        else:
            outputs.append(read[0] * gate_value)
    return torch.stack(outputs, dim=1), state


_FACTORIZED_INPUTS = 18


class _TritonFactorizedQuaternionRead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        aw, ax, ay, az,
        qw, qx, qy, qz,
        kw, kx, ky, kz,
        vw, vx, vy, vz,
        inject, gate,
        full_read, chunk_size,
    ):
        inputs = (
            aw, ax, ay, az, qw, qx, qy, qz, kw, kx, ky, kz,
            vw, vx, vy, vz, inject, gate,
        )
        if any(not tensor.is_cuda for tensor in inputs) and not _TRITON_INTERPRET:
            raise RuntimeError("factorized Triton quaternion read requires CUDA tensors")
        if any(tensor.dtype != torch.float32 or not tensor.is_contiguous() for tensor in inputs):
            raise ValueError("factorized Triton quaternion read expects contiguous FP32 buffers")

        B, T, H, K = aw.shape
        V = vw.shape[3]
        if T <= 0:
            raise ValueError("sequence length must be positive")
        chunk_size = max(1, min(int(chunk_size), T))
        n_chunks = (T + chunk_size - 1) // chunk_size
        checkpoints = tuple(
            torch.empty((B,n_chunks+1,H,V,K), dtype=torch.float32, device=aw.device)
            for _ in range(4)
        )
        output_shape = (B,T,H,V,4) if full_read else (B,T,H,V)
        output = torch.empty(output_shape, dtype=torch.float32, device=aw.device)
        block_k = triton.next_power_of_2(K)
        _factorized_read_fwd[(B * H * V,)](
            *inputs,
            *checkpoints,
            output,
            T=T, H=H, V=V, K=K,
            N_CHUNKS=n_chunks, CHUNK_SIZE=chunk_size,
            BLOCK_K=block_k, FULL_READ=bool(full_read),
            num_warps=4 if block_k >= 32 else 1,
        )

        ctx.save_for_backward(*inputs, *checkpoints)
        ctx.full_read = bool(full_read)
        ctx.chunk_size = chunk_size
        ctx.n_chunks = n_chunks
        ctx.output_shape = output_shape
        ctx.set_materialize_grads(False)
        final_state = tuple(checkpoint[:,-1].contiguous() for checkpoint in checkpoints)
        return output, *final_state

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_statew, grad_statex, grad_statey, grad_statez):
        saved = ctx.saved_tensors
        inputs = saved[:_FACTORIZED_INPUTS]
        checkpoints = saved[_FACTORIZED_INPUTS:]
        B, T, H, K = inputs[0].shape
        V = inputs[12].shape[3]

        if grad_output is None:
            grad_output = torch.zeros(ctx.output_shape, dtype=torch.float32, device=inputs[0].device)
        else:
            grad_output = grad_output.contiguous().float()
        final_grads = (grad_statew, grad_statex, grad_statey, grad_statez)
        adjoint = tuple(
            torch.zeros((B,H,V,K), dtype=torch.float32, device=inputs[0].device)
            if grad is None else grad.contiguous().float().clone()
            for grad in final_grads
        )
        grads_a = tuple(torch.empty_like(tensor) for tensor in inputs[0:4])
        grads_q = tuple(torch.empty_like(tensor) for tensor in inputs[4:8])
        grads_k = tuple(torch.empty_like(tensor) for tensor in inputs[8:12])
        # V and gate are reduced over K with atomics; initialize them once.
        grads_v = tuple(torch.zeros_like(tensor) for tensor in inputs[12:16])
        grad_inject = torch.empty_like(inputs[16])
        grad_gate = torch.zeros_like(inputs[17])
        input_grads = (*grads_a, *grads_q, *grads_k, *grads_v, grad_inject, grad_gate)
        temporary = tuple(
            torch.empty(
                (B,ctx.chunk_size,H,V,K),
                dtype=torch.float32,
                device=inputs[0].device,
            )
            for _ in range(4)
        )
        block_k = triton.next_power_of_2(K)
        block_v = triton.next_power_of_2(V)

        for chunk in range(ctx.n_chunks - 1, -1, -1):
            start = chunk * ctx.chunk_size
            stop = min(start + ctx.chunk_size, T)
            chunk_length = stop - start
            _recompute_chunk_states[(B * H * V,)](
                *inputs[0:4],
                *inputs[8:12],
                *inputs[12:16],
                inputs[16],
                *checkpoints,
                *temporary,
                T=T, H=H, V=V, K=K,
                N_CHUNKS=ctx.n_chunks, CHUNK_INDEX=chunk,
                START=start, C_LEN=chunk_length, TEMP_C=ctx.chunk_size,
                BLOCK_K=block_k, num_warps=4 if block_k >= 32 else 1,
            )
            _factorized_chunk_bwd[(B * H * K,)](
                *inputs,
                *checkpoints,
                *temporary,
                grad_output,
                *adjoint,
                *input_grads,
                T=T, H=H, V=V, K=K,
                N_CHUNKS=ctx.n_chunks, CHUNK_INDEX=chunk,
                START=start, C_LEN=chunk_length, TEMP_C=ctx.chunk_size,
                BLOCK_V=block_v, FULL_READ=ctx.full_read,
                num_warps=4 if block_v >= 32 else 1,
            )

        tensor_grads = tuple(
            gradient if needed else None
            for gradient, needed in zip(input_grads, ctx.needs_input_grad[:_FACTORIZED_INPUTS])
        )
        return (*tensor_grads, None, None)


class _TritonQuaternionScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, aw, ax, ay, az, bw, bx, by, bz):
        if not (aw.is_cuda and bw.is_cuda) and not _TRITON_INTERPRET:
            raise RuntimeError("Triton quaternion scan requires CUDA tensors")
        B, T, H, _, K = aw.shape
        V = bw.shape[3]
        inputs = (aw, ax, ay, az, bw, bx, by, bz)
        if any(x.dtype != torch.float32 or not x.is_contiguous() for x in inputs):
            raise ValueError("Triton quaternion scan expects contiguous FP32 buffers")
        outs = tuple(torch.empty_like(bw) for _ in range(4))
        _scan_fwd[(B * H * V * K,)](
            aw, ax, ay, az, bw, bx, by, bz, *outs,
            T=T, H=H, V=V, K=K,
        )
        ctx.save_for_backward(aw, ax, ay, az, *outs)
        ctx.shape = (B, T, H, V, K)
        return outs

    @staticmethod
    def backward(ctx, gw, gx, gy, gz):
        aw, ax, ay, az, ow, ox, oy, oz = ctx.saved_tensors
        B, T, H, V, K = ctx.shape
        grads_out = tuple(g.contiguous().float() for g in (gw, gx, gy, gz))
        grads_a = tuple(torch.empty_like(aw) for _ in range(4))
        grads_b = tuple(torch.empty_like(ow) for _ in range(4))
        block_v = triton.next_power_of_2(V)
        _scan_bwd[(B * H * K,)](
            aw, ax, ay, az, ow, ox, oy, oz, *grads_out,
            *grads_a, *grads_b, T=T, H=H, V=V, K=K,
            BLOCK_V=block_v, num_warps=4 if block_v >= 32 else 1,
        )
        return (*grads_a, *grads_b)


def triton_quaternion_scan(a4, b4):
    """Run the legacy materializing scan with an atomics-free backward.

    This compatibility API still returns the complete state history.  New code
    should prefer :func:`triton_factorized_quaternion_read`.
    """
    tensors = tuple(x.contiguous().float() for x in (*a4, *b4))
    return _TritonQuaternionScan.apply(*tensors)


def _automatic_chunk_size(length):
    target = max(1, int(math.sqrt(length)))
    lower = 1 << max(0, target.bit_length() - 1)
    upper = lower << 1
    chunk = lower if target - lower <= upper - target else upper
    return min(length, max(16, min(512, chunk)))


def triton_factorized_quaternion_read(
    a4,
    q4,
    k4,
    v4,
    inject,
    gate,
    *,
    full_quaternion_read=False,
    chunk_size=0,
):
    """Fuse factorized HGSS recurrence, query read, gate, and sparse checkpoints.

    Parameters use real quaternion components:

    - ``a4``: ``[B,T,H,1,K]`` or ``[B,T,H,K]``
    - ``q4`` and ``k4``: ``[B,T,H,K]``
    - ``v4``: ``[B,T,H,V]``
    - ``inject``: ``[B,T,H,K]``
    - ``gate``: ``[B,T,H,V]`` (already activated)

    Returns ``(mixed, final_state4)``. ``mixed`` is ``[B,T,H,V]`` in scalar
    mode and ``[B,T,H,V,4]`` in full-quaternion mode.  Recurrence and sparse
    boundary states remain FP32. Backward replays only one chunk at a time, so
    the large state term scales as ``O((T/C + C) * H*V*K)`` rather than
    ``O(T*H*V*K)``.
    """
    if len(a4) != 4 or len(q4) != 4 or len(k4) != 4 or len(v4) != 4:
        raise ValueError("a4, q4, k4, and v4 must contain four real components")

    normalized_a = []
    for component in a4:
        if component.ndim == 5:
            if component.shape[3] != 1:
                raise ValueError("a4's value dimension must be singleton")
            component = component.squeeze(3)
        if component.ndim != 4:
            raise ValueError("a4 components must have shape [B,T,H,1,K] or [B,T,H,K]")
        normalized_a.append(component)

    tensors = tuple(
        tensor.contiguous().float()
        for tensor in (*normalized_a, *q4, *k4, *v4, inject, gate)
    )
    aw = tensors[0]
    B, T, H, K = aw.shape
    if min(B,T,H,K) <= 0:
        raise ValueError("B, T, H, and K must be positive")
    expected_key_shape = (B,T,H,K)
    if any(tensor.shape != expected_key_shape for tensor in tensors[:12]):
        raise ValueError("a4, q4, and k4 components must share [B,T,H,K]")
    V = tensors[12].shape[3]
    if V <= 0:
        raise ValueError("V must be positive")
    expected_value_shape = (B,T,H,V)
    if any(tensor.shape != expected_value_shape for tensor in tensors[12:16]):
        raise ValueError("v4 components must share [B,T,H,V]")
    if tensors[16].shape != expected_key_shape:
        raise ValueError("inject must have shape [B,T,H,K]")
    if tensors[17].shape != expected_value_shape:
        raise ValueError("gate must have shape [B,T,H,V]")
    if any(tensor.device != aw.device for tensor in tensors):
        raise ValueError("all factorized Triton inputs must be on the same device")
    if any(not tensor.is_cuda for tensor in tensors) and not _TRITON_INTERPRET:
        raise RuntimeError("factorized Triton quaternion read requires CUDA tensors")

    chunk_size = _automatic_chunk_size(T) if int(chunk_size) <= 0 else min(int(chunk_size), T)
    mixed, *final_state = _TritonFactorizedQuaternionRead.apply(
        *tensors,
        bool(full_quaternion_read),
        chunk_size,
    )
    return mixed, tuple(final_state)

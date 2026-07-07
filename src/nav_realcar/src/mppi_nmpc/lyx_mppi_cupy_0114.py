# 2026.1.4，从整个控制空间加噪改为增量空间加噪
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np            # host arrays
import cupy as cp             # GPU arrays / kernels

################################################################################
# 1. Configuration helper
################################################################################


@dataclass
class Config:
    """Tunables (mirrors original Numba version)."""

    # ───────── Planning horizon ──────────
    horizon_MPPI: int = 100                # timesteps (N)
    dt: float = 0.1                   # seconds per step
    num_samples: int = 1024           # K – number of random rollouts per iteration
    num_iterations: int = 8           # optimisation loops per solve()

    # ───────── Control noise (std‑dev) ──
    noise_sigma_v: float = 0.30       # m/s
    noise_sigma_w: float = 0.10       # rad/s

    # ───────── Control bounds ───────────
    v_min: float = 0.1
    v_max: float = 0.5
    w_min: float = -0.5
    w_max: float = 0.5

    # ───────── Increment limits (Δu) ────
    # 单步线速度/角速度变化上限，用于在增量空间限幅
    max_delta_v: float = 0.2        # m/s per step (≈ 2 m/s^2 when dt=0.1)
    max_delta_w: float = 0.4        # rad/s per step

    # ───────── Cost tuning  ─────────────
    lambda_: float = 0.1              # MPPI temperature β (lower ⇒ greedier)
    dist_weight: float = 1.0          # distance‑to‑goal term weight
    safety_weight: float = 0.5        # 安全权重，值越大越关注安全
    smooth_v_weight: float = 0.0      # 控制线速度平滑
    smooth_w_weight: float = 0.0       # 控制角速度平滑，通常比 v 大
    # 新增：μ/ν 懲罰權重
    w_mu: float = 0.5                # 對 traction_v(≈μ) 的懲罰權重
    w_nu: float = 0.5                # 對 traction_w(≈ν) 的懲罰權重

    # ───────── Map parameters ───────────
    map_resolution: float = 1.0       # metres per cell (square grid)
    map_origin_x: float = 0.0         # 地图原点X坐标
    map_origin_y: float = 0.0         # 地图原点Y坐标

    # ───────── Misc – reproducibility ──
    seed: Optional[int] = None        # RNG seed (None ⇒ random)

################################################################################
# 2. MPPI implementation (CuPy + RawKernel)
################################################################################


def _as_f32(x: float) -> np.float32:
    """Convenience: ensure Python *float32* literal for RawKernel scalars."""
    return np.float32(x)


class MPPI_CuPy:
    """Model‑Predictive Path Integral planner that lives **entirely on the GPU**."""

    alias = "cupy"  # for `from ... import MPPI`

    # .....................................................................
    def __init__(
        self,
        traction_map: np.ndarray | cp.ndarray,
        safety_map: Optional[np.ndarray | cp.ndarray] = None,
        cfg: Optional[Config] = None,
        *,
        device: str | int | None = None,
    ) -> None:
        # ───── device context ────────────────────────────────────────────
        self._dev_ctx = cp.cuda.Device(device) if device is not None else None
        if self._dev_ctx is not None:
            self._dev_ctx.__enter__()

        self.cfg: Config = cfg or Config()
        rng_seed = self.cfg.seed

        # ───── maps to GPU (float32, C‑contig) ───────────────────────────
        self.traction_map = cp.asarray(traction_map, dtype=cp.float32, order="C")
        assert self.traction_map.ndim == 3 and self.traction_map.shape[2] == 2, "traction_map must have shape (H, W, 2)"
        self.safety_map = cp.asarray(safety_map, dtype=cp.float32, order="C")
        assert self.safety_map.shape == self.traction_map.shape[:2], "safety_map must have same H×W as traction_map"

        # ───── pre‑flatten maps for coalesced gathers ────────────────────
        self.H, self.W = self.traction_map.shape[:2]
        self.traction_v = self.traction_map[:, :, 0].ravel()
        self.traction_w = self.traction_map[:, :, 1].ravel()
        self.safety_flat = self.safety_map.ravel()

        # ───── MPPI array sizes ──────────────────────────────────────────
        self.N = self.cfg.horizon_MPPI
        self.K = self.cfg.num_samples

        # Control sequence (mean value) ––kept on GPU the entire time
        self.u_seq = cp.zeros((self.N, 2), dtype=cp.float32)
        self.u_seq[:, 0] = 0.2  # 先假设 0.2 m/s 前进
        self.u_seq[:, 1] = 0.0  # 不转弯

        # 用于调试/可视化的参考轨迹（不再用于 NMPC）
        self.ref_states = None      # MPPI生成的参考状态序列
        self.ref_controls = None    # MPPI生成的参考控制序列

        # Temporary buffers ------------------------------------------------
        self._noise_buf = cp.empty((self.N, self.K, 2), dtype=cp.float32)
        self._control_samples = cp.empty_like(self._noise_buf)
        self._state_buf = cp.empty((self.N + 1, self.K, 3), dtype=cp.float32)
        self._cost_buf = cp.empty(self.K, dtype=cp.float32)
        self._weights = cp.empty(self.K, dtype=cp.float32)

        # RNG – Philox
        self.rng = cp.random.RandomState(rng_seed)

        # ───── compile rollout kernel once (CUDA C) ─────────────────────–
        self._build_rollout_kernel()

    # ------------------------------------------------------------------
    # Public API (same as original)
    # ------------------------------------------------------------------
    def setup(self) -> None:
        pass  # kept for API parity

    # ..................................................................
    def solve(
        self,
        state0: Tuple[float, float, float],
        goal: Tuple[float, float],
    ) -> np.ndarray:
        """Optimise & return control sequence (host array, N×2)."""
        state0_gpu = cp.asarray(state0, dtype=cp.float32)
        goal_gpu = cp.asarray(goal, dtype=cp.float32)

        # MPPI优化
        for _ in range(self.cfg.num_iterations):
            self._sample_noise()
            self._form_candidates()
            self._rollout_kernel_launch(state0_gpu, goal_gpu)
            self._update_sequence()  # soft‑min on GPU

        # 保存MPPI生成的参考轨迹（完整长度），便于可视化或调试
        self.ref_states = cp.asnumpy(self._state_buf[:, 0, :])  # (N+1, 3)
        self.ref_controls = cp.asnumpy(self.u_seq)              # (N, 2)

        # 纯MPPI：直接返回整条控制序列
        return self.ref_controls

    # ..................................................................
    def shift_and_update(self, n: int = 1) -> None:
        if n <= 0:
            return
        self.u_seq[:-n] = self.u_seq[n:]
        self.u_seq[-n:] = 0.0
        
        # # 同时更新参考轨迹
        # if self.ref_states is not None:
        #     self.ref_states[:-n] = self.ref_states[n:]
        # if self.ref_controls is not None:
        #     self.ref_controls[:-n] = self.ref_controls[n:]

    # ..................................................................
    def get_state_rollout(
        self,
        state0: Tuple[float, float, float],
        goal: Tuple[float, float],
        *,
        num_extra: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return optimal & random trajectories (host)."""
        # Note: caller must have run solve() beforehand.
        opt = cp.asnumpy(self._state_buf[:, 0, :])
        rnd = cp.asnumpy(self._state_buf[:, 1 : 1 + num_extra, :]).transpose(1, 0, 2)
        return opt, rnd

    ############################################################################
    # 3. Internal helpers (all GPU except scalar extraction)
    ############################################################################

    # =============== 3.1 noise & candidates ==============================
    def _sample_noise(self) -> None:
        self._noise_buf[:, :, 0] = self.rng.normal(
            0.0, self.cfg.noise_sigma_v, size=(self.N, self.K)
        )
        self._noise_buf[:, :, 1] = self.rng.normal(
            0.0, self.cfg.noise_sigma_w, size=(self.N, self.K)
        )

    # .................................................................
    def _form_candidates(self) -> None:
        """在增量空间采样控制序列，并对 Δu 做限幅。

        对每条轨迹 k：
        - t=0 以当前均值控制 u_seq[0] 为基准，加噪后裁剪到 [v_min, v_max]×[w_min, w_max]
        - t>0 在 Δu 空间加噪并限幅：u_t = u_{t-1} + clip(Δu, [-max_delta, max_delta])
        最后再对整个序列做一次控制边界裁剪。
        """

        # 第 0 个时刻：以当前均值控制为基准，加一次噪声
        self._control_samples[0, :, :] = self.u_seq[0, :][None, :, ]
        self._control_samples[0, :, :] += self._noise_buf[0, :, :]

        # clip 到控制物理边界
        cp.clip(self._control_samples[0, :, 0], self.cfg.v_min, self.cfg.v_max,
            out=self._control_samples[0, :, 0])
        cp.clip(self._control_samples[0, :, 1], self.cfg.w_min, self.cfg.w_max,
            out=self._control_samples[0, :, 1])

        # 后续时刻：在 Δu 空间加噪，并对 Δu 做限幅
        max_dv = self.cfg.max_delta_v
        max_dw = self.cfg.max_delta_w

        for t in range(1, self.N):
            # 上一时刻控制 u_{t-1}，形状 (K, 2)
            prev = self._control_samples[t - 1, :, :]

            # 噪声视作 Δu 噪声，形状 (K, 2)
            du = self._noise_buf[t, :, :]

            # 对 Δu 分别在 v、w 方向做硬限幅
            cp.clip(du[:, 0], -max_dv, max_dv, out=du[:, 0])
            cp.clip(du[:, 1], -max_dw, max_dw, out=du[:, 1])

            # u_t = u_{t-1} + Δu
            self._control_samples[t, :, :] = prev + du

        # 最后一遍整体裁剪，确保所有控制都在物理边界内
        cp.clip(self._control_samples[:, :, 0], self.cfg.v_min, self.cfg.v_max,
            out=self._control_samples[:, :, 0])
        cp.clip(self._control_samples[:, :, 1], self.cfg.w_min, self.cfg.w_max,
            out=self._control_samples[:, :, 1])

    # =============== 3.2 rollout (RawKernel) =============================
    def _build_rollout_kernel(self) -> None:
        """Compile the CUDA C kernel that integrates all trajectories."""
        code = r"""
        extern "C" __global__
        void rollout_kernel(
            const float* __restrict__ control,
            const float* __restrict__ traction_v,
            const float* __restrict__ traction_w,
            const float* __restrict__ safety,
            float* __restrict__ states,
            float* __restrict__ costs,
            const int N, const int K, const int H, const int W,
            const float res, const float dt,
            const float dist_w, const float safety_w,
            const float smooth_v_w, const float smooth_w_w,
            // 新增：μ/ν 懲罰權重
            const float w_mu, const float w_nu,
            const float goal_x, const float goal_y,
            const float x0, const float y0, const float th0,
            const float map_origin_x, const float map_origin_y)
        {
            const int k = blockDim.x * blockIdx.x + threadIdx.x;  // trajectory idx
            if (k >= K) return;

            // --- initial state ---
            float x = x0;
            float y = y0;
            float th = th0;

            // store t=0
            int idx_state = k * 3;
            states[idx_state + 0] = x;
            states[idx_state + 1] = y;
            states[idx_state + 2] = th;

            float cost = 0.0f;
            const float eps = 1e-6f;
            const float safety_factor = safety_w;  // 安全影响因子

            // 上一时刻的控制，用于计算 Δu
            int idx_ctrl0 = k * 2;  // t=0 时刻
            float v_prev = control[idx_ctrl0 + 0];
            float w_prev = control[idx_ctrl0 + 1];

            for (int t = 0; t < N; ++t)
            {
                // control index (time‑major)
                int idx_ctrl = (t * K + k) * 2;
                float v = control[idx_ctrl + 0];
                float w = control[idx_ctrl + 1];

                // map cell (nearest)
                float world_x = x;
                float world_y = y;
                float rel_x = world_x - map_origin_x;
                float rel_y = world_y - map_origin_y;
                int gx = (int)roundf(rel_x / res);
                int gy = (int)roundf(rel_y / res);
                gx = max(0, min(W - 1, gx));
                gy = max(0, min(H - 1, gy));
                int cell = gy * W + gx;

                // safety ∈ [0,1], 值越大越安全，构造风险 proxy = 1 - safety
                float safety_val = safety[cell];
                float risk_proxy = 1.0f - safety_val;

                float v_eff = v * traction_v[cell];
                float w_eff = w * traction_w[cell];

                // integrate (Euler)
                x += v_eff * cosf(th) * dt;
                y += v_eff * sinf(th) * dt;
                th += w_eff * dt;

                // store state t+1
                idx_state = ((t + 1) * K + k) * 3;
                states[idx_state + 0] = x;
                states[idx_state + 1] = y;
                states[idx_state + 2] = th;

                // stage cost
                float dx = x - goal_x;
                float dy = y - goal_y;
                float dist = sqrtf(dx * dx + dy * dy + 1e-12f);
                // 1) 距离惩罚项
                cost += dist_w * dist * dt;
                // 2) 安全惩罚项
                cost += safety_w * risk_proxy * dt;
                // 3) μ/ν 懲罰項：希望 μ、ν 越接近 1 越好
                cost += w_mu * (1.0f - traction_v[cell]);
                cost += w_nu * (1.0f - traction_w[cell]);
                // 4) 控制增量惩罚项 (Δu 平滑项)
                if (t > 0) {
                    float dv = v - v_prev;
                    float dw = w - w_prev;
                    cost += smooth_v_w * dv * dv;
                    cost += smooth_w_w * dw * dw;
                }

                // 更新上一时刻控制
                v_prev = v;
                w_prev = w;
            }

            // terminal cost
            float dxT = x - goal_x;
            float dyT = y - goal_y;
            float distT = sqrtf(dxT * dxT + dyT * dyT + 1e-12f);
            int idx_ctrl_last = ((N - 1) * K + k) * 2;
            float v_last = fabsf(control[idx_ctrl_last]);
            cost += distT / (v_last + eps);

            costs[k] = cost;
        }
        """
        self._rollout_kernel = cp.RawKernel(code, "rollout_kernel")

    # ..................................................................
    def _rollout_kernel_launch(self, state0: cp.ndarray, goal: cp.ndarray) -> None:
        """Launch the RawKernel after **extracting scalars to host**."""
        N, K = self.N, self.K
        threads = 128
        blocks = (K + threads - 1) // threads

        # flatten views (no copy)
        control = self._control_samples.reshape(-1)
        states = self._state_buf.reshape(-1)
        costs = self._cost_buf

        # --- scalar extraction (fix for invalid pointer‑as‑float bug) ----
        goal_x = _as_f32(float(goal[0].item()))
        goal_y = _as_f32(float(goal[1].item()))
        x0 = _as_f32(float(state0[0].item()))
        y0 = _as_f32(float(state0[1].item()))
        th0 = _as_f32(float(state0[2].item()))
        map_origin_x = _as_f32(self.cfg.map_origin_x)
        map_origin_y = _as_f32(self.cfg.map_origin_y)

        # launch kernel
        self._rollout_kernel(
            (blocks,), (threads,),
            (
                control,
                self.traction_v, self.traction_w, self.safety_flat,
                states, costs,
                cp.int32(N), cp.int32(K), cp.int32(self.H), cp.int32(self.W),
                _as_f32(self.cfg.map_resolution), _as_f32(self.cfg.dt),
                _as_f32(self.cfg.dist_weight), _as_f32(self.cfg.safety_weight),
                _as_f32(self.cfg.smooth_v_weight), _as_f32(self.cfg.smooth_w_weight),
                _as_f32(self.cfg.w_mu), _as_f32(self.cfg.w_nu),
                goal_x, goal_y,
                x0, y0, th0,
                map_origin_x, map_origin_y,
            )
        )
    
    # =============== 3.3 MPPI weight update ==============================
    def _update_sequence(self) -> None:
        J = self._cost_buf
        beta = 1.0 / self.cfg.lambda_
        J_min = cp.min(J)
        self._weights = cp.exp(-beta * (J - J_min))
        self._weights /= cp.sum(self._weights)

        # ū ← ū + Σ wᵢ ϵᵢ  (broadcast weight vector over time & control‑dim)
        dw = self._weights[None, :, None]  # shape (1,K,1)
        delta = cp.sum(dw * self._noise_buf, axis=1)  # (N,2)
        self.u_seq += delta

        # clip bounds again
        cp.clip(self.u_seq[:, 0], self.cfg.v_min, self.cfg.v_max, out=self.u_seq[:, 0])
        cp.clip(self.u_seq[:, 1], self.cfg.w_min, self.cfg.w_max, out=self.u_seq[:, 1])

    ############################################################################
    # 4. Context‑manager housekeeping
    ############################################################################
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dev_ctx is not None:
            self._dev_ctx.__exit__(exc_type, exc_val, exc_tb)
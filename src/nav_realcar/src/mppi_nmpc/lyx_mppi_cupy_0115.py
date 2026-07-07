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

    # ───────── Map parameters ───────────
    map_resolution: float = 0.1       # metres per cell (square grid)
    map_origin_x: float = -2.0         # 地图原点X坐标
    map_origin_y: float = -5.0         # 地图原点Y坐标

    # ───────── NMPC-style weights ────────
    # 狀態/終端/控制二次型權重
    q_pos: float = 1.0               # 位置誤差權重 (x, y)
    q_theta: float = 0.1             # 航向誤差權重
    qN_pos: float = 2.0              # 終端位置誤差權重
    qN_theta: float = 0.2            # 終端航向誤差權重
    r_v: float = 0.01                # 控制參考 (線速) 誤差權重
    r_w: float = 0.01                # 控制參考 (角速) 誤差權重

    # μ/ν 與 safety 懲罰權重
    w_mu: float = 0.5                # 對 traction_v(≈μ) 的懲罰權重
    w_nu: float = 0.5                # 對 traction_w(≈ν) 的懲罰權重
    w_safety: float = 0.5            # 對 (1 - safety) 的額外懲罰

    # ───────── Misc – reproducibility ──
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
        self.u_seq[:, 0] = 0.2
        self.u_seq[:, 1] = 0.0

        # 參考狀態/控制，用於NMPC形式代價 (上一輪最優軌跡)
        self.x_ref = cp.zeros((self.N + 1, 3), dtype=cp.float32)  # (N+1,3)
        self.u_ref = cp.zeros((self.N, 2), dtype=cp.float32)      # (N,2)

        # 用于调试/可视化的参考轨迹（不再用于 NMPC）
        self.ref_states = None
        self.ref_controls = None

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

        # 若有上一輪最優軌跡，可作為本輪的參考
        # 這裡簡單使用上一輪 k_best 轨迹的狀態作為 x_ref，u_seq 作為 u_ref
        if self.ref_states is not None and self.ref_controls is not None:
            self.x_ref[...] = cp.asarray(self.ref_states, dtype=cp.float32)
            self.u_ref[...] = cp.asarray(self.ref_controls, dtype=cp.float32)
        else:
            # 初始時沒有參考，使用當前狀態+恆定控制作為平凡參考
            self.x_ref[0, :] = state0_gpu
            for t in range(1, self.N + 1):
                self.x_ref[t, :] = self.x_ref[t - 1, :]
            self.u_ref[...] = self.u_seq

        # MPPI优化
        for _ in range(self.cfg.num_iterations):
            self._sample_noise()
            self._form_candidates()
            self._rollout_kernel_launch(state0_gpu, goal_gpu)
            self._update_sequence()

        # 以代價最小的軌跡作為新的參考狀態軌跡
        k_best = int(cp.argmin(self._cost_buf).get())
        self.ref_states = cp.asnumpy(self._state_buf[:, k_best, :])  # (N+1,3)
        self.ref_controls = cp.asnumpy(self.u_seq)                   # (N,2)

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
        #define M_PI 3.14159265358979323846f
        extern "C" __global__
        void rollout_kernel(
            const float* __restrict__ control,   // N*K*2 (row‑major, time‑major)
            const float* __restrict__ traction_v,
            const float* __restrict__ traction_w,
            const float* __restrict__ safety,
            const float* __restrict__ x_ref,     // (N+1)*3, time-major
            const float* __restrict__ u_ref,     // N*2, time-major
            float* __restrict__ states,          // (N+1)*K*3, time‑major
            float* __restrict__ costs,
            const int N, const int K, const int H, const int W,
            const float res, const float dt,
            const float q_pos, const float q_theta,
            const float qN_pos, const float qN_theta,
            const float r_v, const float r_w,
            const float w_mu, const float w_nu, const float w_safety,
            const float goal_x, const float goal_y,  // 當前局部目標（base_link系）
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

            // 上一時刻控制，用於計算 Δu（雖然當前代價不再顯式使用平滑項，但保留以便擴展）
            int idx_ctrl0 = k * 2;  // t=0 時刻
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

                // traction & safety
                float mu = traction_v[cell];   // 對應論文中的 μ(px,py)
                float nu = traction_w[cell];   // 對應論文中的 ν(px,py)
                float safety_val = safety[cell];
                float risk_proxy = 1.0f - safety_val;  // safety 越小風險越大

                float v_eff = v * mu;
                float w_eff = w * nu;

                // integrate (Euler)
                x += v_eff * cosf(th) * dt;
                y += v_eff * sinf(th) * dt;
                th += w_eff * dt;

                // store state t+1
                idx_state = ((t + 1) * K + k) * 3;
                states[idx_state + 0] = x;
                states[idx_state + 1] = y;
                states[idx_state + 2] = th;

                // ---------- stage cost (NMPC style) ----------
                // 1) 狀態誤差 ||x_t - x_ref_t||_Q^2
                int idx_xref = t * 3;
                float xr = x_ref[idx_xref + 0];
                float yr = x_ref[idx_xref + 1];
                float thr = x_ref[idx_xref + 2];

                float ex = x - xr;
                float ey = y - yr;
                float eth = th - thr;
                // wrap 航向誤差到 [-pi, pi]
                if (eth > M_PI)  eth -= 2.0f * M_PI;
                if (eth < -M_PI) eth += 2.0f * M_PI;

                cost += q_pos * (ex * ex + ey * ey) + q_theta * (eth * eth);

                // 2) 控制誤差 ||u_t - u_ref_t||_R^2
                int idx_uref = t * 2;
                float v_ref = u_ref[idx_uref + 0];
                float w_ref = u_ref[idx_uref + 1];
                float ev = v - v_ref;
                float ew = w - w_ref;
                cost += r_v * ev * ev + r_w * ew * ew;

                // 3) μ/ν 懲罰 (論文中是 -Wμ μ - Wν ν，這裡換成常數 - 獎勵 => 懲罰)
                cost += w_mu * (1.0f - mu);
                cost += w_nu * (1.0f - nu);

                // 4) safety 懲罰 (1 - safety)
                cost += w_safety * risk_proxy;

                // 更新上一時刻控制
                v_prev = v;
                w_prev = w;
            }

            // ---------- terminal cost：改為懲罰終端狀態到當前局部目標的距離 ----------
            // 原來是：x_ref[N] 作為終端參考
            // 現在直接用 goal_x, goal_y 作為期望終端位置
            float dxg = x - goal_x;
            float dyg = y - goal_y;
            float exN = dxg;
            float eyN = dyg;
            // 若需要終端朝向約束，可在此引入期望航向，暫時不約束
            float ethN = 0.0f;

            cost += qN_pos * (exN * exN + eyN * eyN) + qN_theta * (ethN * ethN);

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

        # --- scalar extraction ---
        x0 = _as_f32(float(state0[0].item()))
        y0 = _as_f32(float(state0[1].item()))
        th0 = _as_f32(float(state0[2].item()))
        map_origin_x = _as_f32(self.cfg.map_origin_x)
        map_origin_y = _as_f32(self.cfg.map_origin_y)

        # 參考軌跡展平
        x_ref_flat = self.x_ref.reshape(-1)
        u_ref_flat = self.u_ref.reshape(-1)

        self._rollout_kernel(
            (blocks,), (threads,),
            (
                control,
                self.traction_v, self.traction_w, self.safety_flat,
                x_ref_flat, u_ref_flat,
                states, costs,
                cp.int32(N), cp.int32(K), cp.int32(self.H), cp.int32(self.W),
                _as_f32(self.cfg.map_resolution), _as_f32(self.cfg.dt),
                _as_f32(self.cfg.q_pos), _as_f32(self.cfg.q_theta),
                _as_f32(self.cfg.qN_pos), _as_f32(self.cfg.qN_theta),
                _as_f32(self.cfg.r_v), _as_f32(self.cfg.r_w),
                _as_f32(self.cfg.w_mu), _as_f32(self.cfg.w_nu), _as_f32(self.cfg.w_safety),
                _as_f32(float(goal[0].item())), _as_f32(float(goal[1].item())),
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

        dw = self._weights[None, :, None]
        delta = cp.sum(dw * self._noise_buf, axis=1)
        self.u_seq += delta

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
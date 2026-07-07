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
    horizon: int = 100                # timesteps (N)
    dt: float = 0.1                   # seconds per step
    num_samples: int = 1024           # K – number of random rollouts per iteration
    num_iterations: int = 8           # optimisation loops per solve()

    # ───────── Control noise (std‑dev) ──
    noise_sigma_v: float = 0.30       # m/s
    noise_sigma_w: float = 0.50       # rad/s

    # ───────── Control bounds ───────────
    v_min: float = -1.0
    v_max: float = 1.0
    w_min: float = -1.5
    w_max: float = 1.5

    # ───────── Cost tuning  ─────────────
    lambda_: float = 1.0              # MPPI temperature β (lower ⇒ greedier)
    dist_weight: float = 1.0          # distance‑to‑goal term weight
    risk_cost: float = 10.0           # multiplier for risk map cell value

    risk_temp_factor: float = 2.0      # 风险温度因子γ (决定风险对λ的影响程度)
    base_lambda: float = 1.0           # 基础温度λ0 (无风险时的默认温度)
    lambda_min: float = 0.1            # 最小温度λ限制
    lambda_max: float = 10.0           # 最大温度λ限制

    # ───────── Map parameters ───────────
    map_resolution: float = 1.0       # metres per cell (square grid)

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

    alias = "cupy"  # for `from ... import MPPI as MPPI`

    # .....................................................................
    def __init__(
        self,
        traction_map: np.ndarray | cp.ndarray,
        risk_map: Optional[np.ndarray | cp.ndarray] = None,
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

        if risk_map is None:
            risk_map = np.zeros(self.traction_map.shape[:2], dtype=np.float32)
        self.risk_map = cp.asarray(risk_map, dtype=cp.float32, order="C")
        assert self.risk_map.shape == self.traction_map.shape[:2], "risk_map must have same H×W as traction_map"

        # ───── pre‑flatten maps for coalesced gathers ────────────────────
        self.H, self.W = self.traction_map.shape[:2]
        self.traction_v = self.traction_map[:, :, 0].ravel()
        self.traction_w = self.traction_map[:, :, 1].ravel()
        self.risk_flat = self.risk_map.ravel()

        # ───── MPPI array sizes ──────────────────────────────────────────
        self.N = self.cfg.horizon
        self.K = self.cfg.num_samples

        # Control sequence (mean value) ––kept on GPU the entire time
        self.u_seq = cp.zeros((self.N, 2), dtype=cp.float32)

        # Temporary buffers ------------------------------------------------
        self._noise_buf = cp.empty((self.N, self.K, 2), dtype=cp.float32)
        self._control_samples = cp.empty_like(self._noise_buf)
        self._state_buf = cp.empty((self.N + 1, self.K, 3), dtype=cp.float32)
        self._cost_buf = cp.empty(self.K, dtype=cp.float32)
        self._avg_risk_buf = cp.empty(self.K, dtype=cp.float32)
        self._weights = cp.empty(self.K, dtype=cp.float32)

        # RNG – Philox
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

        for _ in range(self.cfg.num_iterations):
            self._sample_noise()
            self._form_candidates()
            self._rollout_kernel_launch(state0_gpu, goal_gpu)
            self._update_sequence()  # soft‑min on GPU

        return cp.asnumpy(self.u_seq)

    # ..................................................................
    def shift_and_update(self, n: int = 1) -> None:
        if n <= 0:
            return
        self.u_seq[:-n] = self.u_seq[n:]
        self.u_seq[-n:] = 0.0

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
        # Broadcast ū → (N, K, 2) into _control_samples, then add noise.
        self._control_samples[:] = self.u_seq[:, None, :]
        self._control_samples += self._noise_buf

        # In‑place bounds clipping
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
            const float* __restrict__ control,   // N*K*2 (row‑major, time‑major)
            const float* __restrict__ traction_v,
            const float* __restrict__ traction_w,
            const float* __restrict__ risk,
            float* __restrict__ states,          // (N+1)*K*3, time‑major
            float* __restrict__ costs,
            float* __restrict__ avg_risks,       // K (新增：每条轨迹的平均风险)
            const int N, const int K, const int H, const int W,
            const float res, const float dt,
            const float dist_w, const float risk_w,
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
            float total_risk = 0.0f;      // 新增：累计风险值
            const float eps = 1e-6f;

            for (int t = 0; t < N; ++t)
            {
                // control index (time‑major)
                int idx_ctrl = (t * K + k) * 2;
                float v = control[idx_ctrl + 0];
                float w = control[idx_ctrl + 1];

                // map cell (nearest) - 考虑地图原点偏移
                float world_x = x;
                float world_y = y;
                // 计算相对于地图原点的坐标
                float rel_x = world_x - map_origin_x;
                float rel_y = world_y - map_origin_y;
                // 转换为地图索引
                int gx = (int)roundf(rel_x / res);
                int gy = (int)roundf(rel_y / res);
                gx = max(0, min(W - 1, gx));
                gy = max(0, min(H - 1, gy));
                int cell = gy * W + gx;

                float risk_val = risk[cell];
                total_risk += risk_val;    // 累加当前步的风险值
                
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

                cost += dist_w * dist * dt;
            }

            // terminal cost
            float dxT = x - goal_x;
            float dyT = y - goal_y;
            float distT = sqrtf(dxT * dxT + dyT * dyT + 1e-12f);
            int idx_ctrl_last = ((N - 1) * K + k) * 2;
            float v_last = fabsf(control[idx_ctrl_last]);
            cost += distT / (v_last + eps);

            costs[k] = cost;
            avg_risks[k] = total_risk / (float)N;  // 存储轨迹平均风险值
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
        avg_risks = self._avg_risk_buf  # 新增：平均风险缓冲区

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
                self.traction_v, self.traction_w, self.risk_flat,
                states, costs, avg_risks,  # 新增：传递平均风险缓冲区
                cp.int32(N), cp.int32(K), cp.int32(self.H), cp.int32(self.W),
                _as_f32(self.cfg.map_resolution), _as_f32(self.cfg.dt),
                _as_f32(self.cfg.dist_weight), _as_f32(self.cfg.risk_cost),
                goal_x, goal_y,
                x0, y0, th0,
                map_origin_x, map_origin_y,
            )
        )

    # =============== 3.3 MPPI weight update ==============================
    def _update_sequence(self) -> None:
        J = self._cost_buf
        
        # 计算轨迹的平均风险
        mean_risk = cp.mean(self._avg_risk_buf)
        
        # 动态调整λ参数 (λ = λ0 * (1 + γ · mean(risk_rollout)))
        dynamic_lambda = self.cfg.base_lambda * (1.0 + self.cfg.risk_temp_factor * mean_risk)
        
        # 将lambda限制在设定的范围内
        dynamic_lambda = max(self.cfg.lambda_min, min(self.cfg.lambda_max, dynamic_lambda))
        
        # 使用动态λ计算权重
        beta = 1.0 / dynamic_lambda
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
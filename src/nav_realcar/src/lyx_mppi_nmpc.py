from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np            # host arrays
import cupy as cp             # GPU arrays / kernels
import casadi as ca          

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
    noise_sigma_w: float = 0.50       # rad/s

    # ───────── Control bounds ───────────
    v_min: float = 0.1
    v_max: float = 0.5
    w_min: float = -0.5
    w_max: float = 0.5

    # ───────── Cost tuning  ─────────────
    lambda_: float = 0.5              # MPPI temperature β (lower ⇒ greedier)
    dist_weight: float = 1.0          # distance‑to‑goal term weight
    safety_weight: float = 0.5        # 安全权重，值越大越关注安全
    smooth_v_weight: float = 0.5       # 控制线速度平滑
    smooth_w_weight: float = 1.0       # 控制角速度平滑，通常比 v 大

    # ───────── Map parameters ───────────
    map_resolution: float = 0.1       # metres per cell (square grid)
    map_origin_x: float = -2.0         # 地图原点X坐标
    map_origin_y: float = -5.0         # 地图原点Y坐标

    # ───────── NMPC参数 ──────────────────
    use_nmpc: bool = True             # 是否使用NMPC进行精确控制
    Q_x: float = 1.0                  # 状态跟踪误差权重-x
    Q_y: float = 1.0                  # 状态跟踪误差权重-y
    Q_th: float = 0.1                 # 状态跟踪误差权重-theta
    R_v: float = 0.1                  # 控制跟踪误差权重-v
    R_w: float = 0.1                  # 控制跟踪误差权重-w
    QN_x: float = 10.0                # 终端状态误差权重-x
    QN_y: float = 10.0                # 终端状态误差权重-y
    QN_th: float = 1.0                # 终端状态误差权重-theta
    horizon_NMPC: int = 40            # NMPC预测时域

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
        self.u_seq[:, 0] = 0.2  # 先假设 0.2 m/s 前进
        self.u_seq[:, 1] = 0.0  # 不转弯

        # ───── NMPC相关变量 ──────────────────────────────────────────────
        self.N_NMPC = self.cfg.horizon_NMPC
        self.ref_states = None      # MPPI生成的参考状态序列
        self.ref_controls = None    # MPPI生成的参考控制序列
        self.nmpc_controls = None   # NMPC优化的控制序列
        self.solver = None          # CasADi求解器

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

        # 第1阶段：MPPI优化
        for _ in range(self.cfg.num_iterations):
            self._sample_noise()
            self._form_candidates()
            self._rollout_kernel_launch(state0_gpu, goal_gpu)
            self._update_sequence()  # soft‑min on GPU

        # 保存MPPI生成的参考轨迹
        full_states = cp.asnumpy(self._state_buf[:, 0, :])  # (N_MPPI+1, 3)
        full_controls = cp.asnumpy(self.u_seq)              # (N_MPPI, 2)
        self.ref_states = full_states[:self.N_NMPC+1, :]           # (N_NMPC+1, 3)
        self.ref_controls = full_controls[:self.N_NMPC, :]         # (N_NMPC, 2)

        # 如果不使用NMPC，直接返回MPPI结果
        if not self.cfg.use_nmpc:
            return cp.asnumpy(self.u_seq)
            
        # 第2阶段：NMPC优化
        try:
            # 延迟初始化NMPC求解器，只在首次使用时创建
            if self.solver is None:
                self._setup_nmpc()
                
            # 使用MPPI生成的轨迹作为参考，求解NMPC问题
            nmpc_controls = self._solve_nmpc(state0, self.ref_states, self.ref_controls)
            self.nmpc_controls = nmpc_controls
            
            # 返回NMPC的第一个控制动作
            return nmpc_controls[0:1]
        except Exception as e:
            print(f"NMPC求解失败: {e}")
            # 求解失败时退回到MPPI控制
            return cp.asnumpy(self.u_seq[0:1])

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
        # Broadcast ū → (N, K, 2) into _control_samples, then add noise.
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
            const float* __restrict__ safety,
            float* __restrict__ states,          // (N+1)*K*3, time‑major
            float* __restrict__ costs,
            const int N, const int K, const int H, const int W,
            const float res, const float dt,
            const float dist_w, const float safety_w,
            const float smooth_v_w, const float smooth_w_w,
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
                // 3) 控制增量惩罚项 (Δu 平滑项)
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
    # 4. NMPC implementation (using CasADi)
    ############################################################################
    def _setup_nmpc(self) -> None:
        """设置NMPC求解器"""
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.numel()

        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        n_controls = controls.numel()

        rhs = ca.vertcat(
            v * ca.cos(theta),  # x_dot
            v * ca.sin(theta),  # y_dot
            omega               # theta_dot
        )
        
        f = ca.Function('f', [states, controls], [rhs], ['x', 'u'], ['x_dot'])
        X = ca.SX.sym('X', n_states, self.N_NMPC + 1)
        U = ca.SX.sym('U', n_controls, self.N_NMPC)
        P = ca.SX.sym('P', n_states + n_states*(self.N_NMPC+1) + n_controls*self.N_NMPC)
        current_state = P[:n_states]

        ref_states_start = n_states
        ref_states_end = ref_states_start + n_states*(self.N_NMPC+1)
        ref_states = ca.reshape(P[ref_states_start:ref_states_end], n_states, self.N_NMPC+1)
        
        ref_controls_start = ref_states_end
        ref_controls_end = ref_controls_start + n_controls*self.N_NMPC
        ref_controls = ca.reshape(P[ref_controls_start:ref_controls_end], n_controls, self.N_NMPC)

        cost = 0
        Q = ca.diag(ca.vertcat(self.cfg.Q_x, self.cfg.Q_y, self.cfg.Q_th))
        R = ca.diag(ca.vertcat(self.cfg.R_v, self.cfg.R_w))
        QN = ca.diag(ca.vertcat(self.cfg.QN_x, self.cfg.QN_y, self.cfg.QN_th))
        
        g = []
        g.append(X[:, 0] - current_state)
        
        # 阶段代价
        for i in range(self.N_NMPC):
            # 1) 状态跟踪误差惩罚项 ||x_i - x^r_i||^2_Q
            state_error = X[:, i] - ref_states[:, i]
            cost += ca.mtimes(ca.mtimes(state_error.T, Q), state_error) * self.cfg.dt
            
            # 2) 控制跟踪误差惩罚项 ||u_i - u^r_i||^2_R
            control_error = U[:, i] - ref_controls[:, i]
            cost += ca.mtimes(ca.mtimes(control_error.T, R), control_error) * self.cfg.dt
            
            # 系统动力学约束
            x_next = X[:, i]
            u = U[:, i]
            k1 = f(x=x_next, u=u)['x_dot']
            x_next_euler = x_next + self.cfg.dt * k1
            g.append(X[:, i+1] - x_next_euler)
        
        # 终端代价：惩罚预测末端的状态与参考末端状态的偏离 ||x_{k+N} - x^r_{k+N}||^2_{QN}
        terminal_error = X[:, self.N_NMPC] - ref_states[:, self.N_NMPC]
        cost += ca.mtimes(ca.mtimes(terminal_error.T, QN), terminal_error)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_bound_frac': 1e-6,
            'ipopt.warm_start_slack_bound_frac': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_states = n_states
        self.n_controls = n_controls
        self.last_solution = None
    
    def _solve_nmpc(self, state0, ref_states, ref_controls):
        """使用MPPI生成的参考轨迹求解NMPC问题"""
        try:
            n_states = 3  # [x, y, theta]
            n_controls = 2  # [v, omega]
            N = self.N_NMPC
            
            x0 = np.array(state0, dtype=np.float64).reshape(-1)
            p = np.zeros(n_states + n_states*(N+1) + n_controls*N)
            p[:n_states] = x0

            ref_states_flat = ref_states.reshape(-1)
            ref_states_start = n_states
            ref_states_end = ref_states_start + len(ref_states_flat)
            p[ref_states_start:ref_states_end] = ref_states_flat

            ref_controls_flat = ref_controls.reshape(-1)
            ref_controls_start = ref_states_end
            ref_controls_end = ref_controls_start + len(ref_controls_flat)
            p[ref_controls_start:ref_controls_end] = ref_controls_flat
            
            x_init = np.zeros((n_states*(N+1) + n_controls*N, 1))
            
            for i in range(N+1):
                idx = i * n_states
                x_init[idx:idx+n_states, 0] = ref_states[i, :]
            
            control_start = n_states*(N+1)
            for i in range(N):
                idx = control_start + i * n_controls
                x_init[idx:idx+n_controls, 0] = ref_controls[i, :]

            if self.last_solution is not None:
                x_init = self.last_solution

            lbx = -np.inf * np.ones_like(x_init)
            ubx = np.inf * np.ones_like(x_init)

            for i in range(N+1):
                idx = i * n_states + 2
                lbx[idx] = -np.pi
                ubx[idx] = np.pi

            for i in range(N):
                v_idx = control_start + i * n_controls
                w_idx = v_idx + 1
                lbx[v_idx] = self.cfg.v_min
                ubx[v_idx] = self.cfg.v_max
                lbx[w_idx] = self.cfg.w_min
                ubx[w_idx] = self.cfg.w_max

            lbg = np.zeros((n_states * (N + 1), 1))
            ubg = np.zeros((n_states * (N + 1), 1))

            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=p
            )

            x_sol = sol['x'].full()
            self.last_solution = x_sol
            control_start = n_states*(N+1)
            u_sol = x_sol[control_start:].reshape(N, n_controls)
            return u_sol
            
        except Exception as e:
            print(f"NMPC求解过程中出错: {e}")
            # 出错时返回MPPI控制序列
            return ref_controls

    ############################################################################
    # 5. Context‑manager housekeeping
    ############################################################################
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dev_ctx is not None:
            self._dev_ctx.__exit__(exc_type, exc_val, exc_tb)
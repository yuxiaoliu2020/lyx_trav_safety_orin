from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np            # host arrays
import cupy as cp             # GPU arrays / kernels
import casadi as ca          # 新增: NMPC求解器

################################################################################
# 1. Configuration helper
################################################################################


@dataclass
class Config:
    """Tunables (mirrors original Numba version)."""

    # ───────── Planning horizon ──────────
    horizon: int = 100                # timesteps (N)
    dt: float = 0.1                   # seconds per step
    num_samples: int = 1024           # K – number of random rollouts per iteration
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

    # ───────── Map parameters ───────────
    map_resolution: float = 1.0       # metres per cell (square grid)
    map_origin_x: float = 0.0         # 地图原点X坐标
    map_origin_y: float = 0.0         # 地图原点Y坐标

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
    W_mu: float = 2.0                 # 可通行性权重-mu
    W_nu: float = 2.0                 # 可通行性权重-nu
    W_risk: float = 5.0               # 风险权重

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

        # ───── NMPC相关变量 ──────────────────────────────────────────────
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
        self.ref_states = cp.asnumpy(self._state_buf[:, 0, :])
        self.ref_controls = cp.asnumpy(self.u_seq)

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
        
        # 同时更新参考轨迹
        if self.ref_states is not None:
            self.ref_states[:-n] = self.ref_states[n:]
        if self.ref_controls is not None:
            self.ref_controls[:-n] = self.ref_controls[n:]

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
            const float* __restrict__ risk,
            float* __restrict__ states,          // (N+1)*K*3, time‑major
            float* __restrict__ costs,
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
            const float eps = 1e-6f;
            const float risk_factor = 0.8f;  // 风险影响因子

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
                float v_eff = v * traction_v[cell] * (1.0f - risk_factor * risk_val);
                float w_eff = w * traction_w[cell] * (1.0f - risk_factor * risk_val);

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
                self.traction_v, self.traction_w, self.risk_flat,
                states, costs,
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
        """设置NMPC求解器，使用CasADi库"""
        # 符号变量：状态 [x, y, theta]
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.numel()
        
        # 符号变量：控制输入 [v, omega]
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        n_controls = controls.numel()
        
        # 系统动力学模型（单轮模型）
        rhs = ca.vertcat(
            v * ca.cos(theta),  # x_dot
            v * ca.sin(theta),  # y_dot
            omega               # theta_dot
        )
        
        # 创建动力学函数
        f = ca.Function('f', [states, controls], [rhs], ['x', 'u'], ['x_dot'])
        
        # 决策变量
        # 预测时域内的状态序列
        X = ca.SX.sym('X', n_states, self.N + 1)
        # 预测时域内的控制序列
        U = ca.SX.sym('U', n_controls, self.N)
        
        # 参数：当前状态、参考状态序列、参考控制序列、可通行性地图信息
        P = ca.SX.sym('P', n_states + n_states*(self.N+1) + n_controls*self.N + 3*self.N)
        
        # 解析参数向量
        current_state = P[:n_states]  # 当前状态
        
        # 参考状态轨迹
        ref_states_start = n_states
        ref_states_end = ref_states_start + n_states*(self.N+1)
        ref_states = ca.reshape(P[ref_states_start:ref_states_end], n_states, self.N+1)
        
        # 参考控制轨迹
        ref_controls_start = ref_states_end
        ref_controls_end = ref_controls_start + n_controls*self.N
        ref_controls = ca.reshape(P[ref_controls_start:ref_controls_end], n_controls, self.N)
        
        # 路径可通行性估计（从MPPI获取）
        trav_start = ref_controls_end
        trav_mu = P[trav_start:trav_start+self.N]
        trav_nu = P[trav_start+self.N:trav_start+2*self.N]
        risk_vals = P[trav_start+2*self.N:trav_start+3*self.N]
        
        # 定义代价函数
        cost = 0
        
        # 状态权重矩阵 Q（对角阵）
        Q = ca.diag(ca.vertcat(self.cfg.Q_x, self.cfg.Q_y, self.cfg.Q_th))
        
        # 控制权重矩阵 R（对角阵）
        R = ca.diag(ca.vertcat(self.cfg.R_v, self.cfg.R_w))
        
        # 终端状态权重矩阵 QN
        QN = ca.diag(ca.vertcat(self.cfg.QN_x, self.cfg.QN_y, self.cfg.QN_th))
        
        # 初始状态约束
        g = []  # 约束向量
        g.append(X[:, 0] - current_state)
        
        # 为预测时域添加约束和代价函数
        for i in range(self.N):
            # 状态跟踪误差项 ||x_i - x^r_i||^2_Q
            state_error = X[:, i] - ref_states[:, i]
            cost += ca.mtimes(ca.mtimes(state_error.T, Q), state_error) * self.cfg.dt
            
            # 控制跟踪误差项 ||u_i - u^r_i||^2_R
            control_error = U[:, i] - ref_controls[:, i]
            cost += ca.mtimes(ca.mtimes(control_error.T, R), control_error) * self.cfg.dt
            
            # 可通行性和风险代价项 -(W_μ·μ + W_ν·ν - W_risk·risk)
            # 注意：负号是为了最大化可通行性，而风险项前的负号变成正号，因为要最小化风险
            trav_cost = -(self.cfg.W_mu * trav_mu[i] + self.cfg.W_nu * trav_nu[i] - self.cfg.W_risk * risk_vals[i]) * self.cfg.dt
            cost += trav_cost
            
            # 系统动力学约束（使用欧拉积分）
            x_next = X[:, i]
            u = U[:, i]
            k1 = f(x=x_next, u=u)['x_dot']
            x_next_euler = x_next + self.cfg.dt * k1
            g.append(X[:, i+1] - x_next_euler)
        
        # 终端代价 ||x_{k+N} - x^r_{k+N}||^2_{QN}
        terminal_error = X[:, self.N] - ref_states[:, self.N]
        cost += ca.mtimes(ca.mtimes(terminal_error.T, QN), terminal_error)
        
        # 创建NLP问题
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        # 求解器选项
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
        
        # 创建求解器
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 保存维度信息，供后续使用
        self.n_states = n_states
        self.n_controls = n_controls
        
        # 保存最近的优化结果，用于暖启动
        self.last_solution = None
    
    def _solve_nmpc(self, state0, ref_states, ref_controls):
        """使用MPPI生成的参考轨迹求解NMPC问题"""
        try:
            n_states = 3  # [x, y, theta]
            n_controls = 2  # [v, omega]
            N = self.N
            
            # 转换初始状态为numpy数组
            x0 = np.array(state0, dtype=np.float64).reshape(-1)
            
            # 准备参数向量
            # 包括：当前状态、参考状态序列、参考控制序列、可通行性估计
            p = np.zeros(n_states + n_states*(N+1) + n_controls*N + 3*N)
            
            # 当前状态
            p[:n_states] = x0
            
            # 参考状态序列
            ref_states_flat = ref_states.reshape(-1)
            ref_states_start = n_states
            ref_states_end = ref_states_start + len(ref_states_flat)
            p[ref_states_start:ref_states_end] = ref_states_flat
            
            # 参考控制序列
            ref_controls_flat = ref_controls.reshape(-1)
            ref_controls_start = ref_states_end
            ref_controls_end = ref_controls_start + len(ref_controls_flat)
            p[ref_controls_start:ref_controls_end] = ref_controls_flat
            
            # 估计路径可通行性
            # 通过查询地图获取MPPI参考路径上的可通行性值
            trav_mu = np.zeros(N)
            trav_nu = np.zeros(N)
            risk_vals = np.zeros(N)
            
            for i in range(N):
                x, y = ref_states[i+1, 0], ref_states[i+1, 1]
                
                # 计算地图索引
                rel_x = x - self.cfg.map_origin_x
                rel_y = y - self.cfg.map_origin_y
                gx = int(round(rel_x / self.cfg.map_resolution))
                gy = int(round(rel_y / self.cfg.map_resolution))
                
                # 边界检查
                gx = max(0, min(self.W - 1, gx))
                gy = max(0, min(self.H - 1, gy))
                
                # 获取可通行性值
                cell = gy * self.W + gx
                trav_mu[i] = self.traction_v.get()[cell]
                trav_nu[i] = self.traction_w.get()[cell]
                risk_vals[i] = self.risk_flat.get()[cell]
            
            # 将可通行性值添加到参数向量
            trav_start = ref_controls_end
            p[trav_start:trav_start+N] = trav_mu
            p[trav_start+N:trav_start+2*N] = trav_nu
            p[trav_start+2*N:trav_start+3*N] = risk_vals
            
            # 为优化变量设置初始猜测值
            # 使用MPPI的结果作为初始猜测值
            x_init = np.zeros((n_states*(N+1) + n_controls*N, 1))
            
            # 状态初始猜测
            for i in range(N+1):
                idx = i * n_states
                x_init[idx:idx+n_states, 0] = ref_states[i, :]
            
            # 控制初始猜测
            control_start = n_states*(N+1)
            for i in range(N):
                idx = control_start + i * n_controls
                x_init[idx:idx+n_controls, 0] = ref_controls[i, :]
            
            # 使用上次的解作为暖启动（如果有）
            if self.last_solution is not None:
                x_init = self.last_solution
            
            # 变量边界
            lbx = -np.inf * np.ones_like(x_init)
            ubx = np.inf * np.ones_like(x_init)
            
            # 状态边界（theta需要限制在[-π,π]范围内）
            for i in range(N+1):
                idx = i * n_states + 2  # theta的索引
                lbx[idx] = -np.pi
                ubx[idx] = np.pi
            
            # 控制边界
            for i in range(N):
                v_idx = control_start + i * n_controls
                w_idx = v_idx + 1
                lbx[v_idx] = self.cfg.v_min
                ubx[v_idx] = self.cfg.v_max
                lbx[w_idx] = self.cfg.w_min
                ubx[w_idx] = self.cfg.w_max
            
            # 约束边界（等式约束）
            lbg = np.zeros((n_states * (N + 1), 1))
            ubg = np.zeros((n_states * (N + 1), 1))
            
            # 求解NLP问题
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=p
            )
            
            # 提取解
            x_sol = sol['x'].full()
            self.last_solution = x_sol  # 保存解，用于下次暖启动
            
            # 提取控制序列
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
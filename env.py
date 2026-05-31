"""
env.py  —  Gym 環境クラス
===========================================================
gymnasium.Env を継承した本体。

実機 / シミュレーターの切り替えは config.py の BACKEND で。
シミュレーター用スタブ (SimCamera / SimSensors / SimCraneX7) は
sim_backend.py に分離してある。実機移行時はそちらを差し替える。
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config, get_config
from reward import PhaseReward, RewardInfo, info_to_dict
from sim_backend import SimCamera, SimSensors, SimCraneX7



# ─────────────────────────────────────────
#  Gym 環境本体
# ─────────────────────────────────────────
class SoftRobotEnv(gym.Env):
    """
    ソフトロボットハンド × xArm — 紙つまみ環境

    観測（obs_mode="multimodal"）:
        "image":   (H, W, C) float32  カメラ画像
        "sensors": (6,) float32       光センサー値

    アクション:
        [Δx, Δy, Δz, grip]  float32  各 [-1, 1]（実際の値は action_limit でスケール）

    報酬 / 終了条件:
        → reward.py 参照
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: Config = None, render_mode: str = None):
        super().__init__()

        self.cfg         = cfg or get_config()
        self.render_mode = render_mode
        self.obs_mode    = self.cfg.obs_mode

        ec = self.cfg.env
        self._action_limit = np.array(ec.action_limit, dtype=np.float32)
        self._dt           = 1.0 / ec.control_hz

        # ── 観測・行動空間の定義 ──────────
        img_space = spaces.Box(
            0.0, 1.0,
            shape=(ec.img_channels, ec.img_height, ec.img_width),  # CHW
            dtype=np.float32,
        )
        sen_space = spaces.Box(0.0, 1.0, shape=(ec.n_sensors,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(ec.action_dim,), dtype=np.float32)

        if self.obs_mode == "camera":
            self.observation_space = img_space
        elif self.obs_mode == "sensor":
            self.observation_space = sen_space
        else:  # multimodal: sensor_hist + joints + phase_id を追加
            hist_flat_dim = ec.sensor_history_len * ec.n_sensors
            joint_space   = spaces.Box(-1.0, 1.0, shape=(ec.n_joints,),
                                       dtype=np.float32)
            phase_space   = spaces.Box(0.0, 1.0, shape=(ec.n_phase_ids,),
                                       dtype=np.float32)  # one-hot [P1,2a,2b,2c,P3]
            joint_vel_space = spaces.Box(-1.0, 1.0, shape=(ec.n_joints,),
                                          dtype=np.float32)
            self.observation_space = spaces.Dict({
                "image":       img_space,
                "sensors":     sen_space,
                "sensor_hist": spaces.Box(0.0, 1.0, shape=(hist_flat_dim,),
                                          dtype=np.float32),
                "joints":      joint_space,
                "joint_vel":   joint_vel_space,
                "phase_id":    phase_space,
            })
        self.action_space = act_space

        # ── ハードウェア初期化 ────────────
        self._init_backend()

        # ── 報酬関数 ──────────────────────
        self._reward_fn = PhaseReward(self.cfg.reward)

        # ── 状態変数 ──────────────────────
        self._phase        = 1
        self._step_count   = 0
        self._sensors_prev = np.zeros(ec.n_sensors, dtype=np.float32)
        self._grip_prev    = 0.0
        self._paper_pos    = np.zeros(3)   # [x, y, z]

        # ── Phase 2 材質適応把持 ──────────
        self._sensor_history     = deque(maxlen=ec.sensor_history_len)
        self._phase2_sub         = 0   # 0=CONTACT, 1=READ, 2=READY
        self._phase2b_steps      = 0   # Phase2b 滞在ステップ数（最低 sensor_history_len 必要）
        self._phase2_cooldown    = False  # 2b 失敗後の再突入禁止フラグ（無限ループ防止）

        # ── 関節速度推定 ──────────────────
        self._joints_prev    = np.zeros(ec.n_joints, dtype=np.float32)  # raw rad
        self._joint_vel_norm = np.zeros(ec.n_joints, dtype=np.float32)  # normalized

        # ── Grip ローパスフィルタ（実機モーター保護: ジャーク制限） ──
        self._grip_state = 0.0   # 実際にモーターへ送られる grip 値（rate-limited）

        # ── ログ ──────────────────────────
        self._episode_reward = 0.0
        self._reward_log: list[RewardInfo] = []

    def _init_backend(self):
        """シミュレーター or 実機の初期化"""
        if self.cfg.backend == "sim":
            self._cam     = SimCamera(self.cfg.env.img_height, self.cfg.env.img_width)
            self._sensors = SimSensors(self.cfg.env.n_sensors,
                                       self.cfg.env.sensor_noise_std)
            self._arm     = SimCraneX7()
        else:
            # ── 実機 CRANE-X7 ───────────────────────────────────
            # import rospy
            # from crane_x7_control.srv import ...
            # self._arm = RealCraneX7(node_name="soft_robot_rl")
            # self._cam = RealCamera(device="/dev/video0")
            # self._sensors = RealSensors(port="/dev/ttyUSB0")
            raise NotImplementedError("実機 CRANE-X7 バックエンドは各自実装してください。")

    # ─────────────────────────────────────
    #  reset
    # ─────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 紙の位置をランダム化（Domain Randomization）
        noise = self.cfg.env.paper_pos_noise
        paper_xy = np.random.uniform(-noise, noise, 2)
        self._paper_pos = np.array([paper_xy[0], paper_xy[1], 0.01])  # z=1cm

        if self.cfg.backend == "sim":
            self._arm.reset()
            self._cam.set_paper_pos(paper_xy)
            # センサーの DR: エピソードごとにヒステリシス係数とドリフトを変更
            self._sensors.reset(
                alpha_range=self.cfg.env.sensor_hysteresis_alpha,
                drift_std=self.cfg.env.sensor_drift_std,
            )

        self._phase        = 1
        self._step_count   = 0
        self._sensors_prev = np.zeros(self.cfg.env.n_sensors, dtype=np.float32)
        self._grip_prev    = 0.0
        self._episode_reward = 0.0
        self._reward_log.clear()
        self._reward_fn.reset()

        self._sensor_history.clear()
        self._phase2_sub      = 0
        self._phase2b_steps   = 0
        self._phase2_cooldown = False
        self._joints_prev     = np.zeros(self.cfg.env.n_joints, dtype=np.float32)
        self._joint_vel_norm  = np.zeros(self.cfg.env.n_joints, dtype=np.float32)
        self._grip_state      = 0.0

        obs  = self._get_obs()
        info = {"phase": 1}
        return obs, info

    # ─────────────────────────────────────
    #  step
    # ─────────────────────────────────────
    def step(self, action: np.ndarray):
        """
        action: (4,) in [-1, 1]
          [Δx_norm, Δy_norm, Δz_norm, grip_norm]
        """
        ec = self.cfg.env

        # ── アクションをスケール ──────────
        a = action * self._action_limit
        dx, dy, dz, grip = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        grip = np.clip(grip, 0.0, 1.0)

        # ── grip 目標値の決定 ──────────────
        # Phase1 と Phase2 cooldown 中はモーターに送る目標値を 0 にする。
        # buffer には Actor の raw action が保存されるため、Critic は
        # 「これらの状態では grip は遷移に寄与しない」と直接学習し、
        # Actor は tanh を飽和させずに済む。
        if self._phase == 1 or self._phase2_cooldown:
            grip_target = 0.0
        else:
            grip_target = grip

        # ── grip ローパス（Action Rate Limiting） ──
        # 実機モーター保護: 1step あたりの変化量を grip_rate_limit に制限。
        # これにより Phase 切替時に grip が 0 → 1 へ瞬時に跳ぶ
        # 「ゴーストグリップ」事故を防ぐ。
        rate = ec.grip_rate_limit
        delta = np.clip(grip_target - self._grip_state, -rate, rate)
        self._grip_state += float(delta)
        grip_applied = float(np.clip(self._grip_state, 0.0, 1.0))

        # ── アーム移動 ────────────────────
        self._arm.move_delta(dx, dy, dz)
        self._arm.set_grip(grip_applied)

        if self.cfg.backend == "sim":
            self._sensors.update_from_grip(grip_applied, self._arm.pos[2], self._paper_pos[2])

        # ── 関節速度計算（差分 / dt / π で正規化）─────
        joints_now = self._arm.get_joint_states()
        self._joint_vel_norm = np.clip(
            (joints_now - self._joints_prev) / (np.pi * self._dt), -1.0, 1.0
        ).astype(np.float32)
        self._joints_prev = joints_now.copy()

        # ── 観測取得 ──────────────────────
        sensors_now = self._sensors.read()
        self._sensor_history.append(sensors_now.copy())   # 材質推定用履歴を更新

        # ── Phase2 cooldown 解除チェック（無限ループ防止） ──
        # 2b 失敗フォールバック後、センサーが十分に下がるまで
        # 2a→2b 再突入を禁止する。grip も上で 0 に強制されている
        # ため数 step で sensors が低下し、自然と解除される。
        if self._phase2_cooldown and sensors_now.max() < ec.phase2_cooldown_thresh:
            self._phase2_cooldown = False

        obs         = self._get_obs()

        # ── 報酬計算 ──────────────────────
        reward, info_r, terminated, truncated = self._compute_reward(
            sensors_now, grip_applied, dz
        )
        self._episode_reward += reward
        self._reward_log.append(info_r)

        # ── 状態更新 ──────────────────────
        self._sensors_prev = sensors_now.copy()
        self._grip_prev    = grip_applied
        self._step_count  += 1

        # ── タイムアウト ──────────────────
        if self._step_count >= ec.max_steps:
            truncated = True

        info = {
            "phase":          self._phase,
            "phase2_sub":     self._phase2_sub if self._phase == 2 else 0,
            "step":           self._step_count,
            "episode_reward": self._episode_reward,
            **info_to_dict(info_r),
        }

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────
    #  報酬計算（フェーズ分岐）
    # ─────────────────────────────────────
    def _compute_reward(self, sensors_now, grip_now, dz):
        ec  = self.cfg.env
        arm = self._arm

        terminated = False
        truncated  = False
        info_r     = RewardInfo()

        if self._phase == 1:
            dist, in_view = self._cam.detect_paper(arm.pos[:2])
            reward, info_r, phase_done = self._reward_fn.phase1(
                dist_xy=dist,
                in_view=in_view,
                phase1_threshold=ec.phase1_done_dist,
            )
            if not in_view:
                terminated = True
            elif phase_done:
                self._sensor_history.clear()   # Phase2a 開始時に Phase1 ゴミ履歴を破棄
                self._phase = 2

        elif self._phase == 2:
            # センサー時系列の安定性を計算（phase2b で使用）
            if len(self._sensor_history) > 2:
                hist_arr   = np.array(list(self._sensor_history))
                sensor_std = float(hist_arr.std(axis=0).mean())
            else:
                sensor_std = 1.0   # 履歴不足時は不安定とみなす

            if self._phase2_sub == 0:   # CONTACT: 紙中心へ移動しながら接触確立
                _, dist_to_center, _ = self._cam.detect_paper_center(arm.pos[:2])
                reward, info_r, sub_done, ep_done = self._reward_fn.phase2a(
                    sensors=sensors_now,
                    sensors_prev=self._sensors_prev,
                    dt=self._dt,
                    contact_thresh=ec.phase2_contact_thresh,
                    dist_to_center=dist_to_center,
                    grip_torque=grip_now,
                    approach_torque=ec.phase2a_approach_torq,
                    center_thresh=ec.phase2a_center_thresh,
                )
                # cooldown 中は 2b 再突入を禁止（無限ループ防止）
                if sub_done and not self._phase2_cooldown:
                    self._phase2_sub    = 1
                    self._phase2b_steps = 0   # READ 入場時にリセット

            elif self._phase2_sub == 1:  # READ: 材質読み取り
                reward, info_r, sub_done, ep_done = self._reward_fn.phase2b(
                    sensors=sensors_now,
                    sensors_prev=self._sensors_prev,
                    sensor_std=sensor_std,
                    dt=self._dt,
                    read_steps=ec.phase2_read_steps,
                    stable_std=ec.phase2_stable_std,
                )
                self._phase2b_steps += 1
                # sensor_history_len ステップ滞在して初めて READY へ（履歴飢餓防止）
                if sub_done and self._phase2b_steps >= ec.sensor_history_len:
                    self._phase2_sub = 2
                # ハードリミット突破 → Reward Hacking 回避: ペナルティを与えて Phase2a へ
                # cooldown フラグを立てて、センサーが下がるまで 2b 再突入を禁止
                # （grip も自動で 0 強制 → 数 step で sensors 低下 → 解除）
                elif self._phase2b_steps >= ec.phase2b_max_steps:
                    timeout_pen = self.cfg.reward.r_phase2b_timeout
                    reward += timeout_pen
                    info_r.r_terminal += timeout_pen
                    info_r.total      += timeout_pen
                    self._phase2_sub      = 0
                    self._phase2b_steps   = 0
                    self._phase2_cooldown = True
                    self._reward_fn.reset_phase2b()

            else:                        # READY: リフト準備確認
                reward, info_r, sub_done, ep_done = self._reward_fn.phase2c(
                    sensors=sensors_now,
                    sensors_prev=self._sensors_prev,
                    dt=self._dt,
                    ready_steps=ec.phase2_ready_steps,
                    grip_thresh=ec.phase2_grip_thresh,
                )
                if sub_done:   # Phase3 へ
                    self._phase = 3
                    self._reward_fn.set_z0(arm.pos[2])

            if ep_done:
                terminated = True

        elif self._phase == 3:
            reward, info_r, success, ep_done = self._reward_fn.phase3(
                arm_z=arm.pos[2],
                sensors=sensors_now,
                sensors_prev=self._sensors_prev,
                action_dz=dz,
                grip_now=grip_now,
                grip_prev=self._grip_prev,
                dt=self._dt,
                lift_target=ec.lift_target_m,
                hold_required=ec.lift_hold_steps,
                slip_thresh=ec.slip_thresh,
            )
            if success:
                terminated = True   # 成功終了
            elif ep_done:
                terminated = True   # 失敗終了（落下）
            elif (sensors_now.min() < ec.phase3_fallback_thresh
                  and sensors_now.max() >= 0.05):
                # センサーが下がったが完全落下ではない → Phase2c に戻して把持立て直し
                self._phase      = 2
                self._phase2_sub = 2
                self._reward_fn.reset_phase2c()

        # ── 特異点・関節限界ペナルティ（全フェーズ共通） ──
        # |joint|/π が singularity_warn を超えると、過剰量に比例した
        # 負の報酬を与える。実機で IK 暴走や関節衝突を起こす姿勢から
        # 早期に離脱するインセンティブになる。
        if self.cfg.backend == "sim":
            limit_ratio = self._arm.get_joint_limit_ratio()
            over = max(0.0, limit_ratio - ec.singularity_warn)
            if over > 0.0:
                sing_pen = self.cfg.reward.r_singularity * over
                reward += sing_pen
                info_r.r_singularity = sing_pen
                info_r.total += sing_pen

        return reward, info_r, terminated, truncated

    # ─────────────────────────────────────
    #  観測取得
    # ─────────────────────────────────────
    def _get_obs(self):
        img     = self._cam.read()
        img_chw = np.transpose(img, (2, 0, 1))    # (H,W,C) → (C,H,W)
        sensors = self._sensors.read()

        if self.obs_mode == "camera":
            return img_chw
        elif self.obs_mode == "sensor":
            return sensors

        # multimodal: sensor_hist を追加
        ec       = self.cfg.env
        hist_len = ec.sensor_history_len
        n_sen    = ec.n_sensors

        if len(self._sensor_history) > 0:
            hist = list(self._sensor_history)
            # エッジパディング: 最初の実値で過去を埋める。
            # ゼロ埋めだと 0→実値の不連続が 1D-CNN にエッジスパイクとして検出され、
            # MaterialEncoder の出力が初期数 step で乱高下する原因になる。
            edge_val = hist[0]
            pad      = [edge_val.copy() for _ in range(hist_len - len(hist))]
            hist_arr = np.array(pad + hist, dtype=np.float32)   # (hist_len, n_sen)
        else:
            hist_arr = np.zeros((hist_len, n_sen), dtype=np.float32)

        # Phase1 のみセンサー履歴をマスク（Phase2a から実データを MaterialEncoder へ渡す）
        in_hist_phase = self._phase >= 2
        hist_flat = hist_arr.flatten() if in_hist_phase else np.zeros(
            ec.sensor_history_len * n_sen, dtype=np.float32)

        joints_norm = (self._arm.get_joint_states() / np.pi).astype(np.float32)
        return {
            "image":       img_chw,
            "sensors":     sensors,
            "sensor_hist": hist_flat,              # (hist_len * n_sen,)
            "joints":      joints_norm,            # (n_joints,) in [-1, 1]
            "joint_vel":   self._joint_vel_norm,   # (n_joints,) in [-1, 1]
            "phase_id":    self._get_phase_id(),   # (n_phase_ids,) one-hot
        }

    # ─────────────────────────────────────
    #  フェーズ ID（one-hot）
    # ─────────────────────────────────────
    def _get_phase_id(self) -> np.ndarray:
        """
        現在のフェーズを one-hot ベクトルで返す。
        index: 0=Phase1, 1=Phase2a, 2=Phase2b, 3=Phase2c, 4=Phase3
        """
        phase_map = {(1, 0): 0, (2, 0): 1, (2, 1): 2, (2, 2): 3, (3, 0): 4}
        sub = self._phase2_sub if self._phase == 2 else 0
        idx = phase_map.get((self._phase, sub), 0)
        v = np.zeros(self.cfg.env.n_phase_ids, dtype=np.float32)
        v[idx] = 1.0
        return v

    # ─────────────────────────────────────
    #  render（デバッグ用）
    # ─────────────────────────────────────
    def render(self):
        if self.render_mode == "rgb_array":
            return (self._cam.read() * 255).astype(np.uint8)
        # "human" モードは matplotlib 等で別途実装

    def close(self):
        pass


# ─────────────────────────────────────────
#  動作確認
# ─────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    env = SoftRobotEnv(cfg)

    obs, info = env.reset()
    print("=== 環境テスト ===")
    print(f"obs_mode : {cfg.obs_mode}")
    if isinstance(obs, dict):
        print(f"image    : {obs['image'].shape}")
        print(f"sensors  : {obs['sensors'].shape}")
    else:
        print(f"obs      : {obs.shape}")

    total_r = 0.0
    for step in range(20):
        action  = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        print(f"  step={step+1:3d}  phase={info['phase']}  r={r:+.3f}  R={total_r:+.3f}"
              + ("  [DONE]" if terminated or truncated else ""))
        if terminated or truncated:
            break

    env.close()
    print("テスト完了")

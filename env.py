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
        # 遅延注入の最大ステップ数。>0 のとき「実行待ちアクション列」を観測に含め、
        # 遅延で隠れ状態になる保留指令を露わにして MDP（Markov性）を保つ。
        self._max_latency  = int(max(ec.sim_latency_steps))

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
            # サブフェーズ遷移カウンタの進捗 [read, ready, lift_hold] を [0,1] で観測。
            # 遷移を決める隠れ状態を明示し、部分観測（POMDP）化を防ぐ。
            progress_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
            obs_dict = {
                "image":         img_space,
                "sensors":       sen_space,
                "sensor_hist":   spaces.Box(0.0, 1.0, shape=(hist_flat_dim,),
                                            dtype=np.float32),
                "joints":        joint_space,
                "joint_vel":     joint_vel_space,
                "phase_id":      phase_space,
                "phase_progress": progress_space,
            }
            # 遅延ON時のみ「実行待ちアクション列」(max_latency × action_dim) を追加。
            # 遅延=0（既定）では付与せず観測は従来どおり（後方互換）。
            if self._max_latency > 0:
                obs_dict["action_queue"] = spaces.Box(
                    -1.0, 1.0, shape=(self._max_latency * ec.action_dim,),
                    dtype=np.float32)
            self.observation_space = spaces.Dict(obs_dict)
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

        # ── 並進指令ローパス + 実機遅延模擬 ──
        self._vel_state = np.zeros(3, dtype=np.float32)  # dx,dy,dz の EMA 状態
        self._latency   = 0                               # 適用遅延ステップ数（reset で DR）
        self._cmd_queue = None                            # 遅延コマンドキュー（deque or None）
        self._action_queue = None                         # 観測用: 実行待ち raw アクション列

        # ── ログ ──────────────────────────
        self._episode_reward = 0.0
        self._reward_log: list[RewardInfo] = []
        self._is_success     = False   # 真の持ち上げ成功フラグ（評価の成功率に使用）

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
        # gym の self.np_random を使い、reset(seed=...) で再現性を担保する
        # （グローバル np.random だと seed 固定しても紙配置が再現しない）。
        noise = self.cfg.env.paper_pos_noise
        paper_xy = self.np_random.uniform(-noise, noise, 2)
        self._paper_pos = np.array([paper_xy[0], paper_xy[1], 0.01])  # z=1cm

        if self.cfg.backend == "sim":
            self._arm.reset()
            self._cam.set_paper_pos(paper_xy)
            # センサーの DR: エピソードごとにヒステリシス係数とドリフトを変更
            self._sensors.reset(
                alpha_range=self.cfg.env.sensor_hysteresis_alpha,
                drift_std=self.cfg.env.sensor_drift_std,
                creep_rate_range=self.cfg.env.sensor_creep_rate,
                creep_gain=self.cfg.env.sensor_creep_gain,
            )

        self._phase        = 1
        self._step_count   = 0
        self._sensors_prev = np.zeros(self.cfg.env.n_sensors, dtype=np.float32)
        self._grip_prev    = 0.0
        self._episode_reward = 0.0
        self._reward_log.clear()
        self._reward_fn.reset()
        self._is_success = False

        self._sensor_history.clear()
        self._phase2_sub      = 0
        self._phase2b_steps   = 0
        self._phase2_cooldown = False
        self._joints_prev     = np.zeros(self.cfg.env.n_joints, dtype=np.float32)
        self._joint_vel_norm  = np.zeros(self.cfg.env.n_joints, dtype=np.float32)
        self._grip_state      = 0.0

        # 並進ローパス状態と、実機遅延 latency をエピソードごとに DR
        self._vel_state = np.zeros(3, dtype=np.float32)
        if self.cfg.backend == "sim":
            lo, hi = self.cfg.env.sim_latency_steps
            self._latency = int(self.np_random.integers(lo, hi + 1)) if hi > 0 else 0
        else:
            self._latency = 0
        self._cmd_queue = (deque([(0.0, 0.0, 0.0, 0.0)] * self._latency, maxlen=self._latency)
                           if self._latency > 0 else None)
        # 観測用アクション窓は max_latency 固定（遅延が DR で変動しても obs 形状は不変）
        if self._max_latency > 0:
            self._action_queue = deque(
                [np.zeros(self.cfg.env.action_dim, dtype=np.float32)] * self._max_latency,
                maxlen=self._max_latency)
        else:
            self._action_queue = None

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
        dx, dy, dz = float(a[0]), float(a[1]), float(a[2])

        # ── 並進指令ローパス（モーター慣性対策） ──
        # 30Hz の小刻みな dx,dy,dz をそのまま送ると CRANE-X7 の PID が追従し切れず
        # 発振する。EMA で滑らかな速度指令に整形する（move_lowpass_beta=0 で無効）。
        beta = ec.move_lowpass_beta
        self._vel_state = (beta * self._vel_state
                           + (1.0 - beta) * np.array([dx, dy, dz], dtype=np.float32))
        dx, dy, dz = (float(self._vel_state[0]), float(self._vel_state[1]),
                      float(self._vel_state[2]))

        # ── Phase2a 降下速度リミッター（遅延オーバシュート対策） ──
        # 接触検知が通信+モーター遅延で数 step 遅れても紙を突き抜けて机に激突
        # しないよう、Phase2a 中は降下量(dz<0)を微速に制限する（env 側で強制）。
        if self._phase == 2 and self._phase2_sub == 0:
            dz = max(dz, -ec.phase2a_descend_limit)

        # grip は raw action[-1,1] を [0,1] 全域へ写像する。
        # 旧実装は action[3] をそのまま clip(0,1) していたため負側の半分
        # （tanh 出力の約半数）が常に 0 に潰れて探索効率を損ねていた。
        grip = float(np.clip((float(action[3]) + 1.0) * 0.5, 0.0, 1.0))

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

        # ── アーム移動（実機遅延の模擬: sim専用 DR） ──
        # _cmd_queue があるとき、latency ステップ前の指令を実際にモーターへ送る。
        # 実機の通信ラグ+モーター応答遅れを再現し、遅延ロバストな方策を学習させる。
        # MaterialEncoder(1D-CNN) はセンサーの位相遅れ波形からこの遅延を読み取れる。
        if self._cmd_queue is not None:
            dx_a, dy_a, dz_a, grip_a = self._cmd_queue[0]        # latency step 前の指令
            self._cmd_queue.append((dx, dy, dz, grip_applied))  # 現指令を投入（最古を破棄）
        else:
            dx_a, dy_a, dz_a, grip_a = dx, dy, dz, grip_applied

        # 観測用: 直近 max_latency 個の raw アクションを保持し、実行待ち指令を
        # 隠れ状態のままにせず obs に露出させる（遅延MDPの状態拡張でMarkov性を回復）。
        if self._action_queue is not None:
            self._action_queue.append(np.asarray(action, dtype=np.float32))

        self._arm.move_delta(dx_a, dy_a, dz_a)
        self._arm.set_grip(grip_a)

        if self.cfg.backend == "sim":
            self._sensors.update_from_grip(grip_a, self._arm.pos[2], self._paper_pos[2])

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

        # ── 報酬計算（フェーズ遷移を含む） ──
        reward, info_r, terminated, truncated = self._compute_reward(
            sensors_now, grip_applied, dz
        )
        self._episode_reward += reward
        self._reward_log.append(info_r)

        # ── 観測取得（報酬計算の後: フェーズ遷移を obs に反映） ──
        # _compute_reward が self._phase / phase_progress を更新するため、obs を
        # 後で取得しないと遷移ステップで phase_id / phase_progress が 1step 遅れる。
        obs = self._get_obs()

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
            "is_success":     self._is_success,
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
            # 粘弾性クリープ対策: 値そのものの std ではなく 1階差分(step増分)の
            # 大きさで「整定」を判定する。クリープは値をゆっくり動かすが step増分は
            # 指数減衰するため、差分ベースなら絶対値ドリフトに惑わされず緩和完了を
            # 検出でき、Phase2b のタイムアウト無限ループを避けられる。
            if len(self._sensor_history) > 2:
                hist_arr   = np.array(list(self._sensor_history))
                sensor_std = float(np.abs(np.diff(hist_arr, axis=0)).mean())
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
                    damage_thresh=ec.damage_thresh,
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
                    damage_thresh=ec.damage_thresh,
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
                    damage_thresh=ec.damage_thresh,
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
                self._is_success = True
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
        img     = self._cam.read(self._arm.pos[:2])   # eye-in-hand: アームxyで紙を描画
        img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (H,W,C) → (C,H,W)
        # Phase2b以降（接触後）は Eye-in-Hand が指・影でオクルージョンするため画像を
        # ゼロマスクし、触覚主導にして視覚ノイズへの過適合（モダリティ崩壊）を防ぐ。
        # Phase2a は紙中心への視覚センタリングに画像が要るのでマスクしない。
        if self.cfg.env.mask_image_after_contact and (
                (self._phase == 2 and self._phase2_sub >= 1) or self._phase == 3):
            img_chw = np.zeros_like(img_chw)
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
        read_p, ready_p, lift_p = self._reward_fn.progress(
            ec.phase2_read_steps, ec.phase2_ready_steps, ec.lift_hold_steps)
        phase_progress = np.array([read_p, ready_p, lift_p], dtype=np.float32)
        obs = {
            "image":        img_chw,
            "sensors":      sensors,
            "sensor_hist":  hist_flat,              # (hist_len * n_sen,)
            "joints":       joints_norm,            # (n_joints,) in [-1, 1]
            "joint_vel":    self._joint_vel_norm,   # (n_joints,) in [-1, 1]
            "phase_id":     self._get_phase_id(),   # (n_phase_ids,) one-hot
            "phase_progress": phase_progress,       # (3,) [read, ready, lift_hold]
        }
        # 遅延ON時のみ実行待ちアクション列を付与（max_latency × action_dim）
        if self._action_queue is not None:
            obs["action_queue"] = np.concatenate(
                list(self._action_queue)).astype(np.float32)
        return obs

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
            return (self._cam.read(self._arm.pos[:2]) * 255).astype(np.uint8)
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

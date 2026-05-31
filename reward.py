"""
reward.py  —  フェーズ別報酬関数
===========================================================
Phase 2 は材質適応把持の3段階サブフェーズで構成される:
  Phase2a: 接触確立 — センサーが接触閾値を超えるまで押し込む
  Phase2b: 材質読み取り — センサー時系列を安定させて材質を推定
  Phase2c: リフト準備確認 — 把持の安定性を確認してPhase3へ

報酬のスケールは config.py の RewardConfig で全部管理しています。
"""

import numpy as np
from dataclasses import dataclass
from config import RewardConfig


@dataclass
class RewardInfo:
    """報酬の内訳を記録する（TensorBoard / wandb で可視化するため）"""
    total:      float = 0.0
    r_dist:     float = 0.0   # Phase1: 距離
    r_align:    float = 0.0   # Phase1: 位置決め
    r_contact:  float = 0.0   # Phase2a: 初接触確立
    r_center:   float = 0.0   # Phase2a: 紙中心合わせ（CRANE-X7）
    r_torque:   float = 0.0   # Phase2a: トルク適切性（CRANE-X7）
    r_stable:   float = 0.0   # Phase2b: センサー安定読み取り
    r_deform:   float = 0.0   # Phase2b: 材質適合変形
    r_ready:    float = 0.0   # Phase2c: リフト準備確認
    r_press:    float = 0.0   # Phase2b: 圧力帯（旧互換フィールド）
    r_balance:  float = 0.0   # Phase2b: 両指バランス
    r_grasp:    float = 0.0   # Phase2c: 把持確立ボーナス
    r_height:   float = 0.0   # Phase3: 高さ
    r_slip:     float = 0.0   # Phase3: 滑り
    r_sim:      float = 0.0   # Phase3: 同時施行
    r_terminal: float = 0.0   # 成功 / 失敗ボーナス
    r_time:     float = 0.0   # タイムペナルティ
    r_grip_pen: float = 0.0   # Phase1: Soft Constraint（grip > 0）ペナルティ
    r_overgrip: float = 0.0   # Phase3: 過剰把持（センサー上限張り付き）ペナルティ
    r_singularity: float = 0.0  # 全Phase共通: 特異点・関節限界近傍ペナルティ


class PhaseReward:
    """
    フェーズ別報酬クラス。

    使い方:
        reward_fn = PhaseReward(cfg.reward)
        reward, info, done = reward_fn.compute(phase, state)
    """

    def __init__(self, cfg: RewardConfig):
        self.c = cfg
        self._lift_hold_count    = 0   # Phase3: 持ち上げ維持の連続カウント
        self._phase2_read_count  = 0   # Phase2b: 安定読み取りカウント
        self._phase2_ready_count = 0   # Phase2c: リフト準備確認カウント
        self._z0 = 0.0                 # Phase3 開始時の z 座標

    def reset(self):
        """エピソード開始時に呼ぶ"""
        self._lift_hold_count    = 0
        self._phase2_read_count  = 0
        self._phase2_ready_count = 0
        self._z0 = 0.0

    def set_z0(self, z: float):
        """Phase3 移行時に呼ぶ（持ち上げ基準点を記録）"""
        self._z0 = z

    def reset_phase2c(self):
        """Phase3→Phase2c フォールバック時に呼ぶ（再把持確認カウンターをリセット）"""
        self._phase2_ready_count = 0

    def reset_phase2b(self):
        """Phase2b 上限突破フォールバック時に呼ぶ（読み取りカウンターをリセット）"""
        self._phase2_read_count = 0

    # ─────────────────────────────────────
    #  Phase 1: アプローチ
    # ─────────────────────────────────────
    def phase1(
        self,
        dist_xy:    float,       # カメラで計算した紙までの水平距離 [m]
        in_view:    bool,        # カメラ視野内に紙があるか
        phase1_threshold: float, # 遷移閾値（EnvConfig から渡す）
    ) -> tuple[float, RewardInfo, bool]:
        """
        Returns
        -------
        reward  : float
        info    : RewardInfo
        phase_done : bool  True なら Phase2 へ移行

        NOTE: grip ペナルティは廃止（tanh 飽和回避）。
              Phase1 中の grip コマンドは env 側で物理的に無視する。
        """
        info = RewardInfo()

        # 視野外に出たら即終了
        if not in_view:
            info.r_terminal = self.c.r_outofview
            info.total = info.r_terminal
            return info.total, info, True   # done=True → エピソード終了

        # 距離ペナルティ（Dense: 毎step遠いほど損する）
        info.r_dist = self.c.r_dist_scale * dist_xy

        # 近接ボーナス（Dense: 閾値 2× 以内で線形増加 → スパイク不連続を排除）
        approach_range = phase1_threshold * 2.0
        if dist_xy < approach_range:
            info.r_align = self.c.r_align_bonus * (1.0 - dist_xy / approach_range)
        phase_done = dist_xy < phase1_threshold

        # タイムペナルティ（遠回りを抑制）
        info.r_time = self.c.r_time_penalty

        info.total = info.r_dist + info.r_align + info.r_time
        return info.total, info, phase_done

    # ─────────────────────────────────────
    #  Phase 2a: 接触確立
    #   センサーが全て contact_thresh を超えるまで押し込む
    # ─────────────────────────────────────
    def phase2a(
        self,
        sensors:         np.ndarray,
        sensors_prev:    np.ndarray,
        dt:              float,
        contact_thresh:  float,
        dist_to_center:  float,    # 紙中心までの距離 [m]（カメラ計測）
        grip_torque:     float,    # 現在のグリップトルク [0, 1]
        approach_torque: float,    # 推奨アプローチトルク
        center_thresh:   float,    # 中心合わせ許容誤差 [m]
    ) -> tuple[float, RewardInfo, bool, bool]:
        """
        CRANE-X7 接触フェーズ:
          1. 紙の中心に XY を合わせながら降下する
          2. 接触前は低トルク（approach_torque 以下）で慎重にアプローチ
          3. 全センサーが contact_thresh を超えたら完了

        Returns: (reward, info, sub_done, episode_done)
          sub_done=True → Phase2b へ移行
        """
        info = RewardInfo()

        if np.abs(sensors - sensors_prev).max() / dt > 0.50:
            info.r_terminal = self.c.r_damage
            info.total = info.r_terminal
            return info.total, info, False, True

        # 紙中心合わせ報酬（Dense: 近いほど高得点）
        info.r_center = self.c.r_center_scale * dist_to_center
        if dist_to_center < center_thresh:
            info.r_center += self.c.r_center_bonus   # 許容誤差内ボーナス

        # トルク適切性報酬（CRANE-X7: 接触前は低トルクが望ましい）
        p_mean = sensors.mean()
        if p_mean < contact_thresh:
            # 未接触: トルクが低いほど良い（慎重なアプローチ）
            if grip_torque <= approach_torque:
                info.r_torque = self.c.r_torque_ok
            else:
                over = grip_torque - approach_torque
                info.r_torque = -self.c.r_torque_ok * over * 4.0

        # 接触確立（全センサーが閾値超え）
        contacted = bool((sensors >= contact_thresh).all())
        if contacted:
            info.r_contact = self.c.r_contact_bonus

        info.r_time = self.c.r_time_penalty
        info.total = info.r_contact + info.r_center + info.r_torque + info.r_time
        return info.total, info, contacted, False

    # ─────────────────────────────────────
    #  Phase 2b: 材質読み取り
    #   センサーを安定させて光反射率パターンから材質特性を推定する。
    #   MaterialEncoder への入力となるセンサー時系列をここで蓄積する。
    #   「どう調整すべきか」はポリシー（MaterialEncoder出力に条件付き）が学習する。
    # ─────────────────────────────────────
    def phase2b(
        self,
        sensors:     np.ndarray,
        sensors_prev: np.ndarray,
        sensor_std:  float,       # 直近履歴の平均標準偏差（env.py で計算済み）
        dt:          float,
        read_steps:  int,
        stable_std:  float,
    ) -> tuple[float, RewardInfo, bool, bool]:
        """
        Returns: (reward, info, sub_done, episode_done)
          sub_done=True → Phase2c へ移行
        """
        info = RewardInfo()

        if np.abs(sensors - sensors_prev).max() / dt > 0.50:
            info.r_terminal = self.c.r_damage
            info.total = info.r_terminal
            return info.total, info, False, True

        if sensors.max() < 0.05:
            info.r_terminal = self.c.r_damage
            info.total = info.r_terminal
            return info.total, info, False, True

        # センサー安定ボーナス: 標準偏差が小さい → 静止して材質を読めている
        if sensor_std < stable_std:
            info.r_stable = self.c.r_stable_bonus
            self._phase2_read_count += 1
        else:
            self._phase2_read_count = 0

        # 変形帯ボーナス（材質依存の最適変形量はポリシーが学習）
        p_mean = sensors.mean()
        if self.c.press_low <= p_mean <= self.c.press_high:
            info.r_deform = self.c.r_deform_good
        elif p_mean > self.c.press_high:
            info.r_deform = self.c.r_press_bad_scale * (p_mean - self.c.press_high)

        # 両指バランス（指A: センサー0-2, 指B: センサー3-5）
        mean_a = sensors[:3].mean()
        mean_b = sensors[3:].mean()
        info.r_balance = self.c.r_balance * (1.0 - abs(mean_a - mean_b))

        sub_done = self._phase2_read_count >= read_steps
        # Phase2b 強制滞在中はタイムペナルティを免除（ボイコット防止）
        info.r_time = self.c.r_phase2b_time_penalty
        info.total = info.r_stable + info.r_deform + info.r_balance + info.r_time
        return info.total, info, sub_done, False

    # ─────────────────────────────────────
    #  Phase 2c: リフト準備確認
    #   把持が安定していることを確認してから Phase3 へ移行する
    # ─────────────────────────────────────
    def phase2c(
        self,
        sensors:     np.ndarray,
        sensors_prev: np.ndarray,
        dt:          float,
        ready_steps: int,
        grip_thresh: float,
    ) -> tuple[float, RewardInfo, bool, bool]:
        """
        Returns: (reward, info, phase_done, episode_done)
          phase_done=True → Phase3 へ移行
        """
        info = RewardInfo()

        if np.abs(sensors - sensors_prev).max() / dt > 0.50:
            info.r_terminal = self.c.r_damage
            info.total = info.r_terminal
            return info.total, info, False, True

        if sensors.max() < 0.05:
            info.r_terminal = self.c.r_damage
            info.total = info.r_terminal
            return info.total, info, False, True

        # 安定グリップ確認: 全センサーが grip_thresh 以上、急変なし
        grip_ok  = bool(sensors.min() >= grip_thresh)
        delta_ok = bool(np.abs(sensors - sensors_prev).max() / dt < 0.10)

        if grip_ok and delta_ok:
            self._phase2_ready_count += 1
        else:
            self._phase2_ready_count = 0

        phase_done = self._phase2_ready_count >= ready_steps
        if phase_done:
            info.r_ready = self.c.r_ready_bonus
            info.r_grasp = self.c.r_grasp_bonus

        info.r_time = self.c.r_time_penalty
        info.total = info.r_ready + info.r_grasp + info.r_time
        return info.total, info, phase_done, False

    # ─────────────────────────────────────
    #  Phase 3: 持ち上げ（同時施行）
    # ─────────────────────────────────────
    def phase3(
        self,
        arm_z:          float,       # 現在の z 座標 [m]
        sensors:        np.ndarray,  # shape (6,)
        sensors_prev:   np.ndarray,
        action_dz:      float,       # 今のアクション Δz（正=上昇）
        grip_now:       float,       # 今のグリップ圧力
        grip_prev:      float,       # 前ステップのグリップ圧力
        dt:             float,
        lift_target:    float,       # 成功判定高さ（EnvConfig）
        hold_required:  int,         # 維持ステップ数（EnvConfig）
        slip_thresh:    float,       # 滑り閾値（EnvConfig）
    ) -> tuple[float, RewardInfo, bool, bool]:
        """
        Returns
        -------
        reward      : float
        info        : RewardInfo
        success     : bool   True なら成功終了
        episode_done: bool   True なら失敗終了
        """
        info = RewardInfo()
        success = False
        episode_done = False

        # ── 落下検知 ──────────────────────
        # センサーが全消灯 → 紙を落とした
        if sensors.max() < 0.05:
            info.r_terminal = self.c.r_drop
            info.total = info.r_terminal
            return info.total, info, False, True

        # ── 滑り検知 ──────────────────────
        # センサー値の変化速度が大きい → 滑り始めている
        slip_rate = np.abs(sensors - sensors_prev).max() / dt
        if slip_rate > slip_thresh:
            info.r_slip = self.c.r_slip_scale * slip_rate

        # ── 高さ報酬 ──────────────────────
        # 紙を持ちながら上昇しているときだけ加算
        height_gain = max(0.0, arm_z - self._z0)
        info.r_height = self.c.r_height_scale * height_gain

        # ── 同時施行ボーナス ───────────────
        # 上昇しながらグリップを維持 or 強化 → 滑り防止行動を明示報酬
        moving_up   = action_dz > 0.001
        gripping_up = grip_now >= grip_prev - 0.01   # 微減は許容
        if moving_up and gripping_up:
            info.r_sim = self.c.r_simultaneous

        # ── 成功判定 ──────────────────────
        if height_gain >= lift_target:
            self._lift_hold_count += 1
        else:
            self._lift_hold_count = 0

        if self._lift_hold_count >= hold_required:
            info.r_terminal = self.c.r_success
            success = True

        # 過剰把持ペナルティ（センサー上限張り付き → 素材・センサー破損リスク）
        over_grip = max(0.0, sensors.mean() - 0.85)
        info.r_overgrip = -self.c.r_overgrip_scale * over_grip

        # タイムペナルティ
        info.r_time = self.c.r_time_penalty

        info.total = (info.r_height + info.r_slip + info.r_sim
                      + info.r_terminal + info.r_time + info.r_overgrip)
        return info.total, info, success, episode_done


# ─────────────────────────────────────────
#  ユーティリティ
# ─────────────────────────────────────────
def info_to_dict(info: RewardInfo) -> dict:
    """TensorBoard / wandb に渡す辞書に変換"""
    return {
        "reward/total":     info.total,
        "reward/dist":      info.r_dist,
        "reward/align":     info.r_align,
        "reward/contact":   info.r_contact,
        "reward/center":    info.r_center,
        "reward/torque":    info.r_torque,
        "reward/stable":    info.r_stable,
        "reward/deform":    info.r_deform,
        "reward/ready":     info.r_ready,
        "reward/balance":   info.r_balance,
        "reward/grasp":     info.r_grasp,
        "reward/height":    info.r_height,
        "reward/slip":      info.r_slip,
        "reward/sim":       info.r_sim,
        "reward/terminal":  info.r_terminal,
        "reward/time":      info.r_time,
        "reward/grip_pen":   info.r_grip_pen,
        "reward/overgrip":   info.r_overgrip,
        "reward/singularity": info.r_singularity,
    }

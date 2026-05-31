"""
sim_backend.py  —  シミュレーター用ハードウェアスタブ
===========================================================
SoftRobotEnv が依存する 3 種類のスタブ:
  - SimCamera   : ランダム画像を返すカメラ
  - SimSensors  : ヒステリシス + ドリフト付き光センサー
  - SimCraneX7  : 7軸アーム（簡易 IK）

実機に移行する際は、このファイルだけを置き換える。
すべて env.py の SoftRobotEnv._init_backend() から呼び出される。
"""

import numpy as np


# ─────────────────────────────────────────
#  カメラ
# ─────────────────────────────────────────
class SimCamera:
    """
    シミュレーター用カメラスタブ。
    MuJoCo 接続前の動作確認用にランダム画像を返す。
    実機では OpenCV カメラに差し替える。
    """
    def __init__(self, height=84, width=84):
        self.height = height
        self.width  = width
        # 紙のシミュレーション位置（ランダム初期化）
        self._paper_xy = np.zeros(2)

    def set_paper_pos(self, xy: np.ndarray):
        self._paper_xy = xy.copy()

    def read(self) -> np.ndarray:
        """RGB 画像を返す（H, W, C），値 [0, 1]"""
        img = np.random.rand(self.height, self.width, 3).astype(np.float32)
        return img

    def detect_paper(self, arm_xy: np.ndarray) -> tuple[float, bool]:
        """
        紙を検出して距離を返す。
        実機では YOLO / HSV マスクなどに差し替える。

        Returns
        -------
        dist    : float  アーム水平位置から紙まで [m]
        in_view : bool   視野内に紙があるか
        """
        dist    = float(np.linalg.norm(arm_xy - self._paper_xy))
        in_view = dist < 0.3   # 30cm 以内なら視野内
        return dist, in_view

    def detect_paper_center(self, arm_xy: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        紙の中心位置とアームとの誤差を返す（Phase2a 中心合わせ用）。
        実機では深度カメラ + 輪郭検出に差し替える。

        Returns
        -------
        center_xy : (2,) アーム座標系での紙中心 [m]
        dist      : float 紙中心までの距離 [m]
        in_view   : bool  視野内か
        """
        center_xy = self._paper_xy.copy()
        dist      = float(np.linalg.norm(arm_xy - center_xy))
        in_view   = dist < 0.3
        return center_xy, dist, in_view


# ─────────────────────────────────────────
#  ソフトセンサー（ヒステリシス + ドリフト）
# ─────────────────────────────────────────
class SimSensors:
    """
    シミュレーター用センサースタブ。
    実機のソフトセンサー（光導波路 / シリコン）の特性を模倣するため、
    α-混合ヒステリシス（応答遅れ）と、ベースラインドリフトを内蔵する。
    """
    def __init__(self, n=6, noise_std=0.01):
        self.n         = n
        self.noise_std = noise_std
        self._base     = np.zeros(n, dtype=np.float32)
        self._hyst_val = np.zeros(n, dtype=np.float32)  # ヒステリシスバッファ
        self._drift    = np.zeros(n, dtype=np.float32)  # ベースラインドリフト
        self._alpha    = 0.5                            # 応答速度（reset でランダム化）
        self._drift_std = 0.0                           # ドリフト強度

    def reset(self, alpha_range: tuple = (0.3, 0.7), drift_std: float = 0.005):
        """
        エピソード開始時に呼ぶ。ヒステリシス係数とドリフト強度を
        各エピソードで Domain Randomization する（Sim-to-Real ロバスト性）。
        """
        self._hyst_val[:] = 0.0
        self._drift[:]    = 0.0
        self._alpha       = float(np.random.uniform(*alpha_range))
        self._drift_std   = float(drift_std)

    def update_from_grip(self, grip_torque: float, arm_z: float, paper_z: float):
        """
        グリッパーが紙に接触しているときのセンサー値を更新。
        ヒステリシス: 目標値に瞬時には到達せず、α 混合で追従する。
        ドリフト    : 1step ごとに微小ランダムウォーク（限界 ±0.05）。
        """
        contact = arm_z <= paper_z + 0.005
        if contact:
            base_val = np.clip(grip_torque, 0, 1)
            target_a = base_val * np.random.uniform(0.85, 1.0, 3)
            target_b = base_val * np.random.uniform(0.85, 1.0, 3)
            target   = np.concatenate([target_a, target_b]).astype(np.float32)
        else:
            target = np.zeros(self.n, dtype=np.float32)

        # ヒステリシス: actual = α·target + (1-α)·prev_actual
        self._hyst_val = self._alpha * target + (1.0 - self._alpha) * self._hyst_val
        # ベースラインドリフト（ランダムウォーク）
        self._drift   += np.random.normal(0, self._drift_std, self.n).astype(np.float32)
        self._drift    = np.clip(self._drift, -0.05, 0.05)
        self._base     = self._hyst_val + self._drift

    def read(self) -> np.ndarray:
        """センサー値を返す。実機では ADC 読み取りに差し替える。"""
        noise = np.random.normal(0, self.noise_std, self.n)
        return np.clip(self._base + noise, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────
#  CRANE-X7（7軸アーム）
# ─────────────────────────────────────────
class SimCraneX7:
    """
    シミュレーター用 CRANE-X7 スタブ（7軸）。
    実機では crane_x7 ROS パッケージ / Dynamixel SDK に差し替える。

    CRANE-X7 仕様:
        関節数: 7（XH430-W350 × 6 + XH540-W270 × 1）
        可動範囲: ±π rad（各軸）
        エンドエフェクタ: 並行グリッパー（gripper=0:開, 1:最大把持力）
    """
    N_JOINTS = 7

    def __init__(self):
        self.pos     = np.zeros(3)    # エンドエフェクタ位置 [x, y, z] [m]
        self.grip    = 0.0            # グリッパートルク [0, 1]
        self._home   = np.array([0.0, 0.0, 0.3])
        self._joints = np.zeros(self.N_JOINTS)  # 関節角度 [rad]

    def reset(self):
        self.pos     = self._home.copy()
        self.grip    = 0.0
        self._joints = np.zeros(self.N_JOINTS)

    def move_delta(self, dx: float, dy: float, dz: float):
        """
        カルテシアン差分移動。
        実機では MoveIt! / crane_x7_description の IK に差し替える。
        """
        self.pos[0] = np.clip(self.pos[0] + dx, -0.5, 0.5)
        self.pos[1] = np.clip(self.pos[1] + dy, -0.5, 0.5)
        self.pos[2] = np.clip(self.pos[2] + dz,  0.0, 0.6)
        # シミュレーター内でIKを近似（7軸を位置から仮計算）
        self._joints = self._approx_ik(self.pos)

    def set_grip(self, torque: float):
        """
        グリッパートルク設定。
        実機では gripper joint の goal_current に差し替える。
        """
        self.grip = float(np.clip(torque, 0.0, 1.0))

    def get_joint_states(self) -> np.ndarray:
        """関節角度（7軸）。実機では Dynamixel の present_position を返す。"""
        return self._joints.copy()

    def get_joint_limit_ratio(self) -> float:
        """
        |joint|/π の最大値（0=中央, 1=限界）。
        sim では _approx_ik の clip(-π, π) に到達する手前で警告に使う。
        実機ではヤコビアン行列式から manipulability を計算するのが本道。
        """
        return float(np.abs(self._joints).max() / np.pi)

    def _approx_ik(self, pos: np.ndarray) -> np.ndarray:
        """7軸の逆運動学近似（シミュレーター専用）。"""
        j = np.zeros(self.N_JOINTS)
        j[0] = np.arctan2(pos[1], pos[0])                    # 旋回
        r    = np.sqrt(pos[0]**2 + pos[1]**2)
        j[1] = np.arctan2(pos[2] - 0.1, r) - np.pi / 4      # 肩
        j[3] = -(j[1] + np.pi / 4)                           # 肘
        return np.clip(j, -np.pi, np.pi)

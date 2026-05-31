"""
encoders.py  —  特徴抽出エンコーダー群
===========================================================
SAC に依存しない、観測 → 潜在ベクトルの変換器を集約。
他の RL アルゴリズム (PPO / TD3 / DreamerV3) に差し替える際も
このファイルはそのまま再利用可能。

含まれるエンコーダー:
  - CameraEncoder      : RGB 画像  → 128 次元
  - SensorEncoder      : 6 センサー → 64 次元
  - MaterialEncoder    : センサー履歴 (20×6) → 16 次元
  - GNNJointEncoder    : 関節 (角度+速度) → 32 次元 (sin/cos エンコード)
  - MultimodalEncoder  : 上記 4 つを concat + PhaseEmbed (32)
"""

import torch
import torch.nn as nn
import numpy as np
from config import NetworkConfig


# ─────────────────────────────────────────
#  カメラ画像エンコーダー（CNN）
# ─────────────────────────────────────────
class CameraEncoder(nn.Module):
    """
    (B, C, H, W) → (B, latent_dim)

    Nature DQN ライクな 3層 CNN。
    画像から紙の位置・姿勢・グリッパーとの位置関係を抽出。
    """

    def __init__(self, cfg: NetworkConfig, img_channels: int = 3):
        super().__init__()

        layers = []
        in_ch = img_channels
        for out_ch, k, s in zip(cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # CNN の出力サイズを自動計算（img_height=84, img_width=84 前提）
        self._cnn_out_dim = self._get_cnn_output_dim(img_channels, 84, 84)

        self.flatten = nn.Flatten()
        self.linear  = nn.Sequential(
            nn.Linear(self._cnn_out_dim, cfg.cnn_latent_dim),
            nn.ReLU(),
        )
        self.latent_dim = cfg.cnn_latent_dim

    def _get_cnn_output_dim(self, c, h, w) -> int:
        dummy = torch.zeros(1, c, h, w)
        with torch.no_grad():
            out = self.cnn(dummy)
        return int(np.prod(out.shape[1:]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, C, H, W)  float32, 値は [0, 1] に正規化済み
        """
        x = self.cnn(img)
        x = self.flatten(x)
        return self.linear(x)          # (B, latent_dim)


# ─────────────────────────────────────────
#  光センサーエンコーダー（MLP）
# ─────────────────────────────────────────
class SensorEncoder(nn.Module):
    """
    (B, n_sensors) → (B, latent_dim)

    6個の光センサー値から指の変形・圧力状態を抽出。
    シンプルな MLP で十分（センサー値は低次元）。
    """

    def __init__(self, cfg: NetworkConfig, n_sensors: int = 6):
        super().__init__()

        layers = []
        in_dim = n_sensors
        for h in cfg.sensor_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, cfg.sensor_latent_dim), nn.ReLU()]

        self.mlp = nn.Sequential(*layers)
        self.latent_dim = cfg.sensor_latent_dim

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        """
        sensors: (B, n_sensors)  float32, 値は [0, 1]
        """
        return self.mlp(sensors)       # (B, latent_dim)


# ─────────────────────────────────────────
#  材質エンコーダー（MaterialEncoder）
# ─────────────────────────────────────────
class MaterialEncoder(nn.Module):
    """
    (B, history_len * n_sensors) → (B, material_dim)

    Phase2b で蓄積したセンサー時系列を 1D-CNN で処理し、材質埋め込みを生成する。
    接触時の光反射率変化カーブが材質を識別する鍵。
    ゼロ入力（Phase1）では null 埋め込みを出力し、ポリシーはこれを無視することを学習する。
    """

    def __init__(self, cfg: NetworkConfig, history_len: int, n_sensors: int):
        super().__init__()
        self.history_len = history_len
        self.n_sensors   = n_sensors

        self.conv = nn.Sequential(
            nn.Conv1d(n_sensors, cfg.material_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(cfg.material_hidden, cfg.material_hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        self.proj = nn.Linear(cfg.material_hidden * 4, cfg.material_dim)
        self.latent_dim = cfg.material_dim

    def forward(self, sensor_hist_flat: torch.Tensor) -> torch.Tensor:
        """
        sensor_hist_flat: (B, history_len * n_sensors)
        Returns: (B, material_dim)
        """
        B = sensor_hist_flat.shape[0]
        x = sensor_hist_flat.view(B, self.history_len, self.n_sensors)
        x = x.permute(0, 2, 1)   # → (B, n_sensors, history_len)
        x = self.conv(x)
        return torch.relu(self.proj(x))


# ─────────────────────────────────────────
#  GNN 関節状態エンコーダー
# ─────────────────────────────────────────
class GNNJointEncoder(nn.Module):
    """
    (B, n_joints) + (B, n_joints) → (B, latent_dim)

    CRANE-X7 の 7 関節を線形運動学チェーンとして扱い、
    親→子の依存関係をメッセージパッシングで学習する。
    PyTorch Geometric 不要 — 純 PyTorch 実装。

    グラフ: 0—1—2—3—4—5—6 (線形運動学チェーン)
    入力 : 関節角度 (sin/cos エンコード) + 関節速度
    """
    N_JOINTS = 7
    EDGES = [
        (0,1),(1,0),(1,2),(2,1),(2,3),(3,2),
        (3,4),(4,3),(4,5),(5,4),(5,6),(6,5),
    ]

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        h = cfg.gnn_hidden

        # 入力: (sin(q), cos(q), q_dot) — ±π 周回不連続を回避
        self.node_embed = nn.Sequential(nn.Linear(3, h), nn.ReLU())
        self.msg_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(h * 2, h), nn.ReLU())
             for _ in range(cfg.gnn_layers)]
        )
        self.proj = nn.Linear(h, cfg.gnn_latent_dim)
        self.latent_dim = cfg.gnn_latent_dim

        src = torch.tensor([e[0] for e in self.EDGES], dtype=torch.long)
        dst = torch.tensor([e[1] for e in self.EDGES], dtype=torch.long)
        self.register_buffer("_src", src)
        self.register_buffer("_dst", dst)

        deg = torch.zeros(self.N_JOINTS)
        deg.scatter_add_(0, dst, torch.ones(len(dst)))
        self.register_buffer("_degree", deg.clamp(min=1).view(1, -1, 1))

    def forward(self, joints: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """
        joints:    (B, n_joints)  float32, 値は [-1, 1]（関節角度 / π）
        joint_vel: (B, n_joints)  float32, 値は [-1, 1]（関節速度 正規化済み）

        ±π 周回不連続を回避するため、関節角度を (sin, cos) の単位円上に
        エンコードしてからメッセージパッシングする（ロボティクス RL 定石）。
        """
        B = joints.shape[0]
        q_rad = joints * float(np.pi)                          # [-π, π] に復元
        node_feat = torch.stack(
            [torch.sin(q_rad), torch.cos(q_rad), joint_vel], dim=-1
        )                                                       # (B, 7, 3)
        x = self.node_embed(node_feat)                          # (B, 7, h)

        for layer in self.msg_layers:
            msg = layer(torch.cat([x[:, self._src],
                                   x[:, self._dst]], dim=-1))   # (B, E, h)
            agg = torch.zeros_like(x)
            agg.scatter_add_(1,
                self._dst.view(1, -1, 1).expand(B, -1, x.shape[-1]), msg)
            x = x + agg / self._degree                    # 残差 + 正規化集約

        return torch.relu(self.proj(x.mean(dim=1)))        # (B, latent_dim)


# ─────────────────────────────────────────
#  マルチモーダル融合エンコーダー
# ─────────────────────────────────────────
class MultimodalEncoder(nn.Module):
    """
    画像・センサー・材質・関節を concat して結合特徴量を作る。

    obs_mode に応じて自動切り替え：
      "camera"     → CameraEncoder のみ
      "sensor"     → SensorEncoder のみ
      "multimodal" → concat(Camera + Sensor + Material + GNNJoint + PhaseEmbed)
    """

    def __init__(self, cfg: NetworkConfig, obs_mode: str,
                 img_channels: int = 3, n_sensors: int = 6,
                 history_len: int = 20, n_joints: int = 7, n_phase: int = 5):
        super().__init__()
        self.obs_mode = obs_mode

        if obs_mode in ("camera", "multimodal"):
            self.cam_enc = CameraEncoder(cfg, img_channels)
        if obs_mode in ("sensor", "multimodal"):
            self.sen_enc = SensorEncoder(cfg, n_sensors)

        if obs_mode == "multimodal":
            self.mat_enc    = MaterialEncoder(cfg, history_len, n_sensors)
            self.gnn_enc    = GNNJointEncoder(cfg)
            self.phase_embed = nn.Sequential(
                nn.Linear(n_phase, cfg.phase_embed_dim), nn.ReLU()
            )

        # 融合後の次元
        if obs_mode == "camera":
            self.latent_dim = cfg.cnn_latent_dim
        elif obs_mode == "sensor":
            self.latent_dim = cfg.sensor_latent_dim
        else:  # multimodal: CNN + Sensor + Material + GNN + PhaseEmbed
            self.latent_dim = (cfg.cnn_latent_dim + cfg.sensor_latent_dim
                               + cfg.material_dim + cfg.gnn_latent_dim
                               + cfg.phase_embed_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        """
        obs: {
            "image":       (B, C, H, W)           ← camera/multimodal
            "sensors":     (B, n_sensors)          ← sensor/multimodal
            "sensor_hist": (B, history_len*n_sen)  ← multimodal のみ
            "joints":      (B, n_joints)           ← multimodal のみ, [-1,1]
            "joint_vel":   (B, n_joints)           ← multimodal のみ, [-1,1]
            "phase_id":    (B, n_phase)            ← multimodal のみ, one-hot
        }
        Returns: (B, latent_dim)
        """
        parts = []
        if self.obs_mode in ("camera", "multimodal"):
            parts.append(self.cam_enc(obs["image"]))
        if self.obs_mode in ("sensor", "multimodal"):
            parts.append(self.sen_enc(obs["sensors"]))
        if self.obs_mode == "multimodal":
            parts.append(self.mat_enc(obs["sensor_hist"]))
            parts.append(self.gnn_enc(obs["joints"], obs["joint_vel"]))
            parts.append(self.phase_embed(obs["phase_id"]))   # one-hot → 32次元埋め込み

        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

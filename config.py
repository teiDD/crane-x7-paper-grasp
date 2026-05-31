"""
config.py  —  全ハイパーパラメータはここだけ触ればOK
===========================================================
VSCode で開いて数値を変えるだけで実験を切り替えられます。
"""

from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────
#  観測モード
#   "camera"     : カメラのみ（アブレーション用）
#   "sensor"     : 光センサーのみ（アブレーション用）
#   "multimodal" : カメラ + 光センサー（本命）
# ─────────────────────────────────────────
OBS_MODE: Literal["camera", "sensor", "multimodal"] = "multimodal"

# シミュレーション or 実機
#   "sim"  : MuJoCo シミュレーター（開発・検証用）
#   "real" : 実機 xArm + 実センサー
BACKEND: Literal["sim", "real"] = "sim"


@dataclass
class EnvConfig:
    # ── 制御 ──────────────────────────────
    control_hz:   int   = 30          # 制御ループ周波数 [Hz]
    max_steps:    int   = 500         # 1エピソードの最大ステップ数
    action_dim:   int   = 4           # [Δx, Δy, Δz, grip]
    action_limit: tuple = (0.05, 0.05, 0.10, 1.0)
    #                                   ↑Δx  ↑Δy   ↑Δz  ↑grip[N·m]

    # ── カメラ ────────────────────────────
    img_height:   int = 84
    img_width:    int = 84
    img_channels: int = 3             # RGB

    # ── 光センサー ─────────────────────────
    n_sensors:     int   = 6          # センサー総数（指A×3 + 指B×3）
    sensor_hz:     int   = 1000       # センサーサンプリング [Hz]
    active_lights: int   = 3          # 点灯数の初期値（0〜6で調整）

    # ── 関節 ──────────────────────────────
    n_joints:      int   = 7          # CRANE-X7 関節数

    # ── フェーズ遷移閾値 ──────────────────
    phase1_done_dist:  float = 0.02   # 紙まで 2cm 以内で Phase2 へ
    phase2_done_hold:  int   = 3      # 把持確立判定の連続ステップ数
    phase2_grip_thresh: float = 0.3   # 把持確立に必要な最低センサー値

    # ── 終了判定 ──────────────────────────
    lift_target_m:  float = 0.05      # 成功に必要な持ち上げ高さ [m]
    lift_hold_steps: int  = 10        # 持ち上げ維持ステップ数
    slip_thresh:    float = 0.05      # 滑り検知閾値（センサー変化速度）
    damage_thresh:  float = 0.50      # 破損検知閾値（急激変化）

    # ── Phase 2 材質適応把持（3段階） ────────
    sensor_history_len:    int   = 20     # 材質推定用センサー履歴長 [steps]
    phase2_contact_thresh: float = 0.10   # 2a→2b: 全センサーが超えたら接触確立
    phase2_read_steps:     int   = 3      # 2b: 安定読み取りに必要なステップ数（緩和: 探索の谷を回避）
    phase2_ready_steps:    int   = 2      # 2c: リフト準備確認ステップ数（緩和）
    phase2_stable_std:     float = 0.05   # センサー安定判定（緩和: 初期探索で達成可能な値）
    phase2b_max_steps:     int   = 50     # 2b 強制脱出ハードリミット（reward hacking 防止）
    phase2_cooldown_thresh: float = 0.05  # Phase2b フォールバック後、再突入を許可するセンサー上限

    # ── センサー Domain Randomization（実機ヒステリシス/ドリフト対策） ──
    sensor_hysteresis_alpha: tuple = (0.3, 0.7)  # α-混合係数の DR 範囲（小=遅い応答）
    sensor_drift_std:        float = 0.005       # ベースラインドリフトのランダムウォーク標準偏差

    # ── アクション安全制御（実機ハードウェア保護） ──
    grip_rate_limit:    float = 0.15      # 1step あたりの grip 変化上限（ジャーク制限）
    singularity_warn:   float = 0.90      # |joint|/π > これでペナルティ開始（特異点近傍）

    # ── フェーズ ID ────────────────────────
    n_phase_ids:           int   = 5      # [Phase1, 2a, 2b, 2c, Phase3] の one-hot 次元数

    # ── CRANE-X7 Phase2a 接触制御 ────────────
    arm_type:              str   = "crane_x7"  # ロボットアーム種別
    phase2a_center_thresh: float = 0.008       # 紙中心への許容誤差 [m] (8mm)
    phase2a_approach_torq: float = 0.25        # 接触前の上限トルク（慎重なアプローチ）
    phase2a_center_xy_lim: float = 0.010       # 2a でのXY微調整幅 [m] (±1cm)

    # ── ランダム化（Domain Randomization）──
    paper_pos_noise:  float = 0.03    # 紙位置のランダム幅 ±3cm
    sensor_noise_std: float = 0.01    # センサーガウスノイズ


@dataclass
class NetworkConfig:
    # ── CNN（カメラ画像エンコーダー）─────
    cnn_channels:    tuple = (32, 64, 64)   # 各畳み込み層のチャンネル数
    cnn_kernels:     tuple = (8, 4, 3)       # カーネルサイズ
    cnn_strides:     tuple = (4, 2, 1)       # ストライド
    cnn_latent_dim:  int   = 128             # CNN出力次元

    # ── センサー MLP ──────────────────────
    sensor_hidden:     tuple = (64, 64)      # 隠れ層サイズ
    sensor_latent_dim: int   = 64            # センサーエンコーダー出力次元

    # ── 融合後 Actor / Critic ─────────────
    # multimodal : 128 + 64 = 192
    # camera only: 128
    # sensor only: 64
    actor_hidden:  tuple = (256, 256)        # Actor の隠れ層
    critic_hidden: tuple = (256, 256)        # Critic の隠れ層（Twin Q）

    # ── 材質エンコーダー（MaterialEncoder） ──
    material_dim:    int = 16   # 材質埋め込みベクトル次元
    material_hidden: int = 64   # 1D-CNN 内部チャンネル数

    # ── GNN（関節状態エンコーダー）────────────
    gnn_hidden:     int = 64    # メッセージパッシング隠れ次元
    gnn_layers:     int = 2     # グラフ畳み込み層数
    gnn_latent_dim: int = 32    # GNN 出力次元

    # ── フェーズ ID エンコーダー ────────────
    phase_embed_dim: int = 32   # one-hot(5) → 32次元埋め込み（直接 concat では窒息）


@dataclass
class SACConfig:
    # ── 基本 ──────────────────────────────
    lr_actor:  float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha:  float = 3e-4           # 温度パラメータの学習率
    gamma:     float = 0.99           # 割引率
    tau:       float = 0.005          # ソフトターゲット更新率

    # ── 探索 ──────────────────────────────
    alpha_init:     float = 0.2       # 初期温度（自動調整される）
    target_entropy: float = -4.0      # 目標エントロピー = -dim(A)

    # ── バッファ・バッチ ───────────────────
    buffer_size:  int = 1_000_000
    batch_size:   int = 256
    warmup_steps: int = 10_000        # この間はランダム行動でバッファ充填
    update_freq:  int = 4             # 何ステップごとにネットワーク更新

    # ── 学習スケジュール ──────────────────
    total_steps:   int = 1_000_000
    eval_freq:     int = 10_000       # 評価頻度（ステップ）
    eval_episodes: int = 5            # 評価時のエピソード数
    save_freq:     int = 50_000       # チェックポイント保存頻度


@dataclass
class RewardConfig:
    # ── Phase 1: アプローチ ───────────────
    r_dist_scale:   float = -0.1      # 距離ペナルティ係数
    r_align_bonus:  float = +2.0      # 位置決め成功ボーナス
    r_time_penalty: float = -0.01     # タイムペナルティ（毎step）
    r_outofview:    float = -5.0      # 視野外ペナルティ

    # ── Phase 2: 把持 ─────────────────────
    r_press_good:      float = +1.5   # 適切圧力帯 [0.3, 0.7]
    r_press_bad_scale: float = -2.0   # 過剰圧力ペナルティ係数
    r_balance:         float = +0.5   # 両指均等ボーナス最大値
    r_grasp_bonus:     float = +3.0   # 把持確立ボーナス（スケール統一）
    r_damage:          float = -5.0   # 破損ペナルティ（スケール統一）
    press_low:         float = 0.3    # 適切圧力帯 下限
    press_high:        float = 0.7    # 適切圧力帯 上限（上げると強く掴む）

    # ── Phase 3: 持ち上げ ─────────────────
    r_height_scale:  float = +2.0     # 高さ報酬係数（m単位）
    r_slip_scale:    float = -3.0     # 滑りペナルティ係数
    r_simultaneous:  float = +0.5     # 同時施行ボーナス（上昇+グリップ増）
    r_success:       float = +10.0    # 最終成功ボーナス（スケール統一: max ±10）
    r_drop:          float = -10.0    # 落下ペナルティ（スケール統一）

    # ── Phase 2 sub-phase（材質適応把持） ─────
    r_contact_bonus: float = +1.0    # Phase2a: 初接触確立ボーナス
    r_stable_bonus:  float = +0.3    # Phase2b: センサー安定読み取り（/step）
    r_deform_good:   float = +1.0    # Phase2b: 材質適合変形帯ボーナス
    r_ready_bonus:   float = +3.0    # Phase2c: リフト準備完了ボーナス

    # ── CRANE-X7 Phase2a 中心合わせ・トルク ──
    r_center_scale:  float = -2.0    # Phase2a: 紙中心ズレペナルティ係数
    r_center_bonus:  float = +1.5    # Phase2a: 許容誤差内に入ったボーナス
    r_torque_ok:     float = +0.5    # Phase2a: 適切トルクボーナス（/step）

    # ── Soft Constraint・Phase別調整 ────────
    r_phase2b_time_penalty: float =  0.0   # Phase2b強制滞在中はタイムペナルティ免除
    r_phase2b_timeout:      float = -3.0   # Phase2b 上限突破（材質読み取り失敗）ペナルティ
    r_overgrip_scale:       float =  2.0   # Phase3: センサー上限張り付き（過剰把持）ペナルティ
    r_singularity:          float = -5.0   # 特異点・関節限界近傍ペナルティ（過剰量×係数）


@dataclass
class Config:
    env:     EnvConfig     = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sac:     SACConfig     = field(default_factory=SACConfig)
    reward:  RewardConfig  = field(default_factory=RewardConfig)

    obs_mode: str = OBS_MODE
    backend:  str = BACKEND

    # ── ログ・出力 ─────────────────────────
    exp_name:  str  = "run_01"        # 実験名（結果フォルダ名になる）
    log_dir:   str  = "logs"
    ckpt_dir:  str  = "checkpoints"
    use_wandb: bool = False           # Weights & Biases ログ（要アカウント）
    seed:      int  = 42


# ─────────────────────────────────────────
#  デフォルト設定を取得するヘルパー
# ─────────────────────────────────────────
def get_config() -> Config:
    return Config()


# ─────────────────────────────────────────
#  アブレーション用プリセット
# ─────────────────────────────────────────
def camera_only_config() -> Config:
    cfg = get_config()
    cfg.obs_mode = "camera"
    cfg.exp_name = "ablation_camera_only"
    return cfg


def sensor_only_config() -> Config:
    cfg = get_config()
    cfg.obs_mode = "sensor"
    cfg.exp_name = "ablation_sensor_only"
    return cfg


def multimodal_config() -> Config:
    cfg = get_config()
    cfg.obs_mode = "multimodal"
    cfg.exp_name = "multimodal_main"
    return cfg

"""
networks.py  —  SAC Actor / Critic ネットワーク
===========================================================
特徴抽出エンコーダーは encoders.py に分離。このファイルは
SAC アルゴリズム特有の Actor (Squashed Gaussian) と
Critic (Twin Q) と、両者を初期化する build_networks() を持つ。

他のアルゴリズム (PPO / TD3) に差し替える際はこのファイルを
書き換え、encoders.py はそのまま再利用する。
"""

import torch
import torch.nn as nn
from config import NetworkConfig
from encoders import MultimodalEncoder


# ─────────────────────────────────────────
#  SAC Actor
# ─────────────────────────────────────────
class SACActorNetwork(nn.Module):
    """
    Squashed Gaussian Actor for SAC。
    出力は (μ, log_σ) → tanh で [-1, 1] にクランプ。
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, encoder: MultimodalEncoder,
                 cfg: NetworkConfig, action_dim: int):
        super().__init__()
        self.encoder = encoder

        in_dim = encoder.latent_dim
        layers = []
        for h in cfg.actor_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.mu_head      = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, obs: dict, detach_encoder: bool = False):
        z = self.encoder(obs)
        if detach_encoder:
            z = z.detach()
        x   = self.mlp(z)
        mu  = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: dict, detach_encoder: bool = False):
        """
        確率的サンプリング（学習時）
        Returns: action (tanh squashed), log_prob
        """
        mu, log_std = self.forward(obs, detach_encoder=detach_encoder)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t  = dist.rsample()                    # reparameterization trick
        y_t  = torch.tanh(x_t)

        # tanh squashing の log_prob 補正
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def act(self, obs: dict, deterministic: bool = False):
        """
        行動出力（推論時）
        deterministic=True → μ そのまま（評価用）
        """
        with torch.no_grad():
            mu, log_std = self.forward(obs)
            if deterministic:
                return torch.tanh(mu)
            std  = log_std.exp()
            x_t  = torch.distributions.Normal(mu, std).sample()
            return torch.tanh(x_t)


# ─────────────────────────────────────────
#  SAC Critic（Twin Q）
# ─────────────────────────────────────────
class SACCriticNetwork(nn.Module):
    """
    Twin Q-network。Q1 と Q2 を同時に計算。
    過大評価バイアスを防ぐため min(Q1, Q2) を使う。
    """

    def __init__(self, encoder: MultimodalEncoder,
                 cfg: NetworkConfig, action_dim: int):
        super().__init__()
        # Critic は obs と action を concat してから MLP
        in_dim = encoder.latent_dim + action_dim

        def make_q():
            layers = []
            d = in_dim
            for h in cfg.critic_hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers.append(nn.Linear(d, 1))
            return nn.Sequential(*layers)

        self.encoder = encoder
        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, obs: dict, action: torch.Tensor):
        z  = self.encoder(obs)
        x  = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: dict, action: torch.Tensor):
        z = self.encoder(obs)
        x = torch.cat([z, action], dim=-1)
        return self.q1(x)


# ─────────────────────────────────────────
#  ファクトリー関数
# ─────────────────────────────────────────
def build_networks(cfg, obs_mode: str, action_dim: int, device: str = "cpu"):
    """
    Actor / Critic / Target Critic を作って返す。

    使い方:
        actor, critic, critic_target = build_networks(cfg, cfg.obs_mode, 4)
    """
    hist_len = cfg.env.sensor_history_len
    n_joints = cfg.env.n_joints
    n_phase  = cfg.env.n_phase_ids
    # 遅延ON時の実行待ちアクション列の次元（max_latency × action_dim）。遅延OFFで0。
    aq_len   = int(max(cfg.env.sim_latency_steps)) * cfg.env.action_dim

    def _make_enc():
        return MultimodalEncoder(cfg.network, obs_mode,
                                 cfg.env.img_channels, cfg.env.n_sensors,
                                 hist_len, n_joints, n_phase,
                                 cfg.env.img_height, cfg.env.img_width,
                                 aq_len).to(device)

    # Actor と Critic は同一エンコーダーを共有（EMA コピー遅延を排除）
    enc    = _make_enc()
    enc_ct = _make_enc()   # Target Critic のみ独立

    actor         = SACActorNetwork(enc,  cfg.network, action_dim).to(device)
    critic        = SACCriticNetwork(enc,  cfg.network, action_dim).to(device)
    critic_target = SACCriticNetwork(enc_ct, cfg.network, action_dim).to(device)

    # ターゲットネットワークは初期値をコピーして固定
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    return actor, critic, critic_target

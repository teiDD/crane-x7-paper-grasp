"""ネットワーク／エンコーダーのスモークテスト

現行 API（GNN は (joints, joint_vel) の2入力、multimodal obs は
joint_vel / phase_id を含む）と形状が一致することを検証する。
実行: python _test_gnn.py
"""
import torch
from config import get_config
from networks import build_networks
from encoders import GNNJointEncoder

cfg = get_config()
nc  = cfg.network
ec  = cfg.env

# ── GNN 単体（joints, joint_vel の2入力）─────────────────
enc = GNNJointEncoder(nc)
joints    = torch.zeros(2, ec.n_joints)
joint_vel = torch.zeros(2, ec.n_joints)
out = enc(joints, joint_vel)
assert out.shape == (2, nc.gnn_latent_dim), out.shape
print(f"GNN output shape : {tuple(out.shape)}  (expected (2, {nc.gnn_latent_dim}))")

# ── multimodal 融合次元の検証 ───────────────────────────
expected_dim = (nc.cnn_latent_dim + nc.sensor_latent_dim
                + nc.material_dim + nc.gnn_latent_dim + nc.phase_embed_dim
                + 3)   # phase_progress (read/ready/lift_hold)
actor, critic, _ = build_networks(cfg, "multimodal", ec.action_dim, "cpu")
assert actor.encoder.latent_dim == expected_dim, (
    actor.encoder.latent_dim, expected_dim)
print(f"latent_dim       : {actor.encoder.latent_dim}  (expected {expected_dim})")

# ── 観測辞書（env._get_obs と同じキー構成）でフォワード ────
hist_dim = ec.sensor_history_len * ec.n_sensors
obs = {
    "image":       torch.zeros(1, ec.img_channels, ec.img_height, ec.img_width),
    "sensors":     torch.zeros(1, ec.n_sensors),
    "sensor_hist": torch.zeros(1, hist_dim),
    "joints":      torch.zeros(1, ec.n_joints),
    "joint_vel":   torch.zeros(1, ec.n_joints),
    "phase_id":    torch.zeros(1, ec.n_phase_ids),
    "phase_progress": torch.zeros(1, 3),
}
feat = actor.encoder(obs)
assert feat.shape == (1, expected_dim), feat.shape
print(f"encoder output   : {tuple(feat.shape)}")

action, log_prob = actor.sample(obs)
assert action.shape == (1, ec.action_dim), action.shape
assert log_prob.shape == (1, 1), log_prob.shape
print(f"action shape     : {tuple(action.shape)}  log_prob: {tuple(log_prob.shape)}")

# ── Critic も通ることを確認 ─────────────────────────────
q1, q2 = critic(obs, action)
assert q1.shape == (1, 1) and q2.shape == (1, 1), (q1.shape, q2.shape)
print(f"critic Q shapes  : {tuple(q1.shape)}, {tuple(q2.shape)}")
print("OK")

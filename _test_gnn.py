"""GNN 実装スモークテスト（使用後削除）"""
import torch
from config import get_config
from networks import build_networks
from encoders import GNNJointEncoder

cfg = get_config()

enc = GNNJointEncoder(cfg.network)
joints = torch.zeros(2, 7)
out = enc(joints)
print(f"GNN output shape : {out.shape}  (expected (2, {cfg.network.gnn_latent_dim}))")

actor, critic, _ = build_networks(cfg, "multimodal", 4, "cpu")
print(f"latent_dim       : {actor.encoder.latent_dim}  (expected {128+64+16+32})")

obs = {
    "image":       torch.zeros(1, 3, 84, 84),
    "sensors":     torch.zeros(1, 6),
    "sensor_hist": torch.zeros(1, 20 * 6),
    "joints":      torch.zeros(1, 7),
}
feat = actor.encoder(obs)
print(f"encoder output   : {feat.shape}")
action, log_prob = actor.sample(obs)
print(f"action shape     : {action.shape}  log_prob: {log_prob.shape}")
print("OK")

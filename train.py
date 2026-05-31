"""
train.py  —  SAC 学習ループ
===========================================================
実行方法:
    python train.py                        # マルチモーダル（デフォルト）
    python train.py --mode camera          # カメラのみ（アブレーション）
    python train.py --mode sensor          # センサーのみ（アブレーション）
    python train.py --exp my_exp_01        # 実験名を指定
    python train.py --seed 123             # シード変更

必要パッケージ:
    pip install torch gymnasium numpy tensorboard
"""

import os
import argparse
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import get_config, camera_only_config, sensor_only_config, multimodal_config
from env import SoftRobotEnv
from networks import build_networks


# ─────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────
class ReplayBuffer:
    """
    辞書形式の観測（multimodal）にも対応した Replay Buffer。
    SAC / TD3 共通で使える。
    """

    def __init__(self, capacity: int, obs_sample, action_dim: int, device: str):
        self.capacity   = capacity
        self.device     = device
        self.action_dim = action_dim
        self._ptr       = 0
        self._size      = 0

        # 観測が dict かどうか
        self._is_dict_obs = isinstance(obs_sample, dict)

        if self._is_dict_obs:
            self._img  = np.zeros((capacity, *obs_sample["image"].shape),  dtype=np.float32)
            self._sen  = np.zeros((capacity, *obs_sample["sensors"].shape), dtype=np.float32)
            self._nimg = np.zeros((capacity, *obs_sample["image"].shape),  dtype=np.float32)
            self._nsen = np.zeros((capacity, *obs_sample["sensors"].shape), dtype=np.float32)
            # sensor_hist / joints は multimodal モードのみ存在する
            self._has_hist   = "sensor_hist" in obs_sample
            self._has_joints = "joints"      in obs_sample
            if self._has_hist:
                self._hist  = np.zeros((capacity, *obs_sample["sensor_hist"].shape), dtype=np.float32)
                self._nhist = np.zeros((capacity, *obs_sample["sensor_hist"].shape), dtype=np.float32)
            if self._has_joints:
                self._jnt  = np.zeros((capacity, *obs_sample["joints"].shape), dtype=np.float32)
                self._njnt = np.zeros((capacity, *obs_sample["joints"].shape), dtype=np.float32)
            self._has_jvel  = "joint_vel" in obs_sample
            if self._has_jvel:
                self._jvel  = np.zeros((capacity, *obs_sample["joint_vel"].shape), dtype=np.float32)
                self._njvel = np.zeros((capacity, *obs_sample["joint_vel"].shape), dtype=np.float32)
            self._has_phase = "phase_id" in obs_sample
            if self._has_phase:
                self._pid  = np.zeros((capacity, *obs_sample["phase_id"].shape), dtype=np.float32)
                self._npid = np.zeros((capacity, *obs_sample["phase_id"].shape), dtype=np.float32)
        else:
            self._obs  = np.zeros((capacity, *obs_sample.shape), dtype=np.float32)
            self._nobs = np.zeros((capacity, *obs_sample.shape), dtype=np.float32)

        self._act  = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rew  = np.zeros((capacity, 1),          dtype=np.float32)
        self._done = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, obs, action, reward, obs_next, done):
        i = self._ptr
        if self._is_dict_obs:
            self._img[i]  = obs["image"]
            self._sen[i]  = obs["sensors"]
            self._nimg[i] = obs_next["image"]
            self._nsen[i] = obs_next["sensors"]
            if self._has_hist:
                self._hist[i]  = obs["sensor_hist"]
                self._nhist[i] = obs_next["sensor_hist"]
            if self._has_joints:
                self._jnt[i]   = obs["joints"]
                self._njnt[i]  = obs_next["joints"]
            if self._has_jvel:
                self._jvel[i]  = obs["joint_vel"]
                self._njvel[i] = obs_next["joint_vel"]
            if self._has_phase:
                self._pid[i]   = obs["phase_id"]
                self._npid[i]  = obs_next["phase_id"]
        else:
            self._obs[i]  = obs
            self._nobs[i] = obs_next

        self._act[i]  = action
        self._rew[i]  = reward
        self._done[i] = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self._size, batch_size)
        d   = self.device

        def t(x):
            return torch.tensor(x[idx], device=d)

        batch = {
            "action":  t(self._act),
            "reward":  t(self._rew),
            "done":    t(self._done),
        }
        if self._is_dict_obs:
            batch["obs"]      = {"image": t(self._img),  "sensors": t(self._sen)}
            batch["obs_next"] = {"image": t(self._nimg), "sensors": t(self._nsen)}
            if self._has_hist:
                batch["obs"]["sensor_hist"]      = t(self._hist)
                batch["obs_next"]["sensor_hist"] = t(self._nhist)
            if self._has_joints:
                batch["obs"]["joints"]           = t(self._jnt)
                batch["obs_next"]["joints"]      = t(self._njnt)
            if self._has_jvel:
                batch["obs"]["joint_vel"]        = t(self._jvel)
                batch["obs_next"]["joint_vel"]   = t(self._njvel)
            if self._has_phase:
                batch["obs"]["phase_id"]         = t(self._pid)
                batch["obs_next"]["phase_id"]    = t(self._npid)
        else:
            batch["obs"]      = t(self._obs)
            batch["obs_next"] = t(self._nobs)

        return batch

    def __len__(self):
        return self._size


# ─────────────────────────────────────────
#  SAC エージェント
# ─────────────────────────────────────────
class SACAgent:

    def __init__(self, cfg, device: str):
        self.cfg    = cfg
        self.device = device
        sc          = cfg.sac
        ec          = cfg.env

        # ── ネットワーク ──────────────────
        self.actor, self.critic, self.critic_target = build_networks(
            cfg, cfg.obs_mode, ec.action_dim, device
        )

        # ── オプティマイザー ───────────────
        # Actor は MLP 部分のみ最適化（エンコーダーは Critic と共有 → Critic 側で更新）
        actor_mlp_params = (
            list(self.actor.mlp.parameters())
            + list(self.actor.mu_head.parameters())
            + list(self.actor.log_std_head.parameters())
        )
        self.opt_actor  = torch.optim.Adam(actor_mlp_params, lr=sc.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=sc.lr_critic)

        # ── 温度パラメータ α（自動調整）────
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha], lr=sc.lr_alpha)
        self.target_ent = sc.target_entropy   # = -dim(A) = -4

        self.gamma = sc.gamma
        self.tau   = sc.tau

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False) -> np.ndarray:
        """観測を受け取って行動を返す（numpy）"""
        obs_t = self._obs_to_tensor(obs)
        action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    def update(self, batch: dict) -> dict:
        """
        1回のネットワーク更新。batch は ReplayBuffer.sample() の出力。
        Returns: ログ用辞書
        """
        sc = self.cfg.sac

        obs      = batch["obs"]
        obs_next = batch["obs_next"]
        action   = batch["action"]
        reward   = batch["reward"]
        done     = batch["done"]

        with torch.no_grad():
            # ターゲット Q 値計算
            next_action, next_log_pi = self.actor.sample(obs_next)
            q1_t, q2_t = self.critic_target(obs_next, next_action)
            min_q_t     = torch.min(q1_t, q2_t) - self.alpha.detach() * next_log_pi
            target_q    = reward + self.gamma * (1 - done) * min_q_t

        # ── Critic 更新 ───────────────────
        q1, q2       = self.critic(obs, action)
        critic_loss  = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # ── Actor 更新 ────────────────────
        new_action, log_pi = self.actor.sample(obs, detach_encoder=True)
        q1_new, q2_new     = self.critic(obs, new_action)
        actor_loss         = (self.alpha.detach() * log_pi - torch.min(q1_new, q2_new)).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # ── 温度 α 更新 ──────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_ent).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # ── ターゲットネットワーク ソフト更新
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "loss/critic": critic_loss.item(),
            "loss/actor":  actor_loss.item(),
            "loss/alpha":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "log_pi_mean": log_pi.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha,
        }, path)
        print(f"  [保存] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
        print(f"  [読込] {path}")

    def _obs_to_tensor(self, obs):
        """numpy 観測 → バッチ付き torch テンソル"""
        if isinstance(obs, dict):
            t_obs = {
                "image":   torch.tensor(obs["image"],   device=self.device).unsqueeze(0),
                "sensors": torch.tensor(obs["sensors"], device=self.device).unsqueeze(0),
            }
            if "sensor_hist" in obs:
                t_obs["sensor_hist"] = torch.tensor(obs["sensor_hist"],
                                                    device=self.device).unsqueeze(0)
            if "joints" in obs:
                t_obs["joints"] = torch.tensor(obs["joints"],
                                               device=self.device).unsqueeze(0)
            if "joint_vel" in obs:
                t_obs["joint_vel"] = torch.tensor(obs["joint_vel"],
                                                  device=self.device).unsqueeze(0)
            if "phase_id" in obs:
                t_obs["phase_id"] = torch.tensor(obs["phase_id"],
                                                  device=self.device).unsqueeze(0)
            return t_obs
        return torch.tensor(obs, device=self.device).unsqueeze(0)


# ─────────────────────────────────────────
#  評価ループ
# ─────────────────────────────────────────
def evaluate(agent: SACAgent, env: SoftRobotEnv, n_episodes: int) -> dict:
    """
    決定論的方策（deterministic=True）で n エピソード実行して成功率を返す。
    学習には影響しない。
    """
    successes   = 0
    total_r     = 0.0
    phase_hist  = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        max_ph = 1

        while not done:
            action          = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r  += r
            done   = term or trunc
            max_ph = max(max_ph, info["phase"])

        total_r    += ep_r
        phase_hist.append(max_ph)
        if max_ph == 3 and ep_r > 0:   # Phase3 に到達して正の報酬 → 成功
            successes += 1

    return {
        "eval/success_rate":  successes / n_episodes,
        "eval/mean_reward":   total_r   / n_episodes,
        "eval/mean_max_phase": np.mean(phase_hist),
    }


# ─────────────────────────────────────────
#  メイン学習ループ
# ─────────────────────────────────────────
def train(cfg):
    # ── 再現性 ────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")
    print(f"obs_mode: {cfg.obs_mode}")
    print(f"exp_name: {cfg.exp_name}")

    # ── ログ・保存ディレクトリ ─────────────
    log_dir  = os.path.join(cfg.log_dir,  cfg.exp_name)
    ckpt_dir = os.path.join(cfg.ckpt_dir, cfg.exp_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ── 環境 ──────────────────────────────
    env      = SoftRobotEnv(cfg)
    env_eval = SoftRobotEnv(cfg)   # 評価用（別インスタンス）

    # ── エージェント ──────────────────────
    agent = SACAgent(cfg, device)

    # ── Replay Buffer ─────────────────────
    obs_sample, _ = env.reset()
    buffer = ReplayBuffer(
        capacity=cfg.sac.buffer_size,
        obs_sample=obs_sample,
        action_dim=cfg.env.action_dim,
        device=device,
    )

    # ── 学習ループ ─────────────────────────
    obs, _          = env.reset()
    ep_reward       = 0.0
    ep_steps        = 0
    ep_count        = 0
    best_success    = 0.0
    recent_rewards  = deque(maxlen=100)   # 直近100エピソードの報酬

    print(f"\n{'─'*55}")
    print(f"  Warm-up: {cfg.sac.warmup_steps:,} steps でバッファ充填中...")
    print(f"{'─'*55}")

    for total_step in range(1, cfg.sac.total_steps + 1):

        # ── 行動選択 ──────────────────────
        if total_step < cfg.sac.warmup_steps:
            action = env.action_space.sample()   # ランダム
        else:
            action = agent.select_action(obs)

        # ── 環境ステップ ──────────────────
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, obs_next, done)
        obs        = obs_next
        ep_reward += reward
        ep_steps  += 1

        # ── エピソード終了処理 ─────────────
        if done:
            ep_count += 1
            recent_rewards.append(ep_reward)

            writer.add_scalar("train/episode_reward", ep_reward,    total_step)
            writer.add_scalar("train/episode_steps",  ep_steps,     total_step)
            writer.add_scalar("train/phase_reached",  info["phase"], total_step)
            writer.add_scalar("train/reward_100ep",
                              np.mean(recent_rewards), total_step)

            # コンソール進捗表示
            if ep_count % 10 == 0:
                print(f"  Ep {ep_count:5d} | step {total_step:8,} "
                      f"| R {ep_reward:+7.2f} | 100avg {np.mean(recent_rewards):+6.2f}"
                      f" | phase {info['phase']}")

            obs, _    = env.reset()
            ep_reward = 0.0
            ep_steps  = 0

        # ── ネットワーク更新 ───────────────
        if total_step >= cfg.sac.warmup_steps and total_step % cfg.sac.update_freq == 0:
            if len(buffer) >= cfg.sac.batch_size:
                batch    = buffer.sample(cfg.sac.batch_size)
                log_dict = agent.update(batch)
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, total_step)

        # ── 評価 ──────────────────────────
        if total_step % cfg.sac.eval_freq == 0:
            eval_dict = evaluate(agent, env_eval, cfg.sac.eval_episodes)
            for k, v in eval_dict.items():
                writer.add_scalar(k, v, total_step)

            sr = eval_dict["eval/success_rate"]
            print(f"\n  ★ 評価 step={total_step:,} | 成功率={sr:.1%}"
                  f" | 平均報酬={eval_dict['eval/mean_reward']:+.2f}\n")

            # ベストモデル保存
            if sr > best_success:
                best_success = sr
                agent.save(os.path.join(ckpt_dir, "best.pt"))

        # ── チェックポイント ───────────────
        if total_step % cfg.sac.save_freq == 0:
            agent.save(os.path.join(ckpt_dir, f"step_{total_step}.pt"))

    print(f"\n学習完了！ best_success_rate = {best_success:.1%}")
    writer.close()
    env.close()
    env_eval.close()


# ─────────────────────────────────────────
#  エントリーポイント
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["camera", "sensor", "multimodal"],
                        default="multimodal", help="観測モード")
    parser.add_argument("--exp",  type=str, default=None, help="実験名")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # モードに応じてコンフィグを選択
    if args.mode == "camera":
        cfg = camera_only_config()
    elif args.mode == "sensor":
        cfg = sensor_only_config()
    else:
        cfg = multimodal_config()

    if args.exp:
        cfg.exp_name = args.exp
    cfg.seed = args.seed

    train(cfg)

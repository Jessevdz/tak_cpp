"""
PPO implementation from Gym: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
"""

import os
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import accumulate
from subprocess import Popen, PIPE


PLAYER_AC_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\players"
OPPONENT_AC_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\opponents"
AC_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\ac.pt"
TMP_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\tmp\\"


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(750, 650),
            nn.ReLU(),
            nn.Linear(650, 550),
            nn.ReLU(),
            nn.Linear(550, 450),
            nn.ReLU(),
            nn.Linear(450, 350),
            nn.ReLU(),
            nn.Linear(350, 1),
        )

    def forward(self, obs):
        return self.v_net(obs)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(750, 800),
            nn.ReLU(),
            nn.Linear(800, 850),
            nn.ReLU(),
            nn.Linear(850, 950),
            nn.ReLU(),
            nn.Linear(950, 1050),
            nn.ReLU(),
            nn.Linear(1050, 1150),
            nn.ReLU(),
            nn.Linear(1150, 1250),
            nn.ReLU(),
            nn.Linear(1250, 1350),
            nn.ReLU(),
            nn.Linear(1350, 1450),
            nn.ReLU(),
            nn.Linear(1450, 1575),
        )

    def sample_action(self, logits):
        # Logits to probabilities
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1, True)
        return action

    def get_action_log_prob(self, logits, act):
        act = act.long().unsqueeze(-1)
        act, log_pmf = torch.broadcast_tensors(act, logits)
        act = act[..., :1]
        return log_pmf.gather(-1, act).squeeze(-1)

    def forward_loss(self, obs, act):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.policy_net(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logp_a = self.get_action_log_prob(logits, act)
        return logp_a

    def forward(self, obs, valid_actions):
        """
        Produce action distributions for given observations
        """
        logits = self.policy_net(obs)
        # Normalize logits: https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        masked_logits = torch.where(valid_actions > 0, logits, torch.tensor([-1e8]))
        action = self.sample_action(masked_logits)

        logp_a = self.get_action_log_prob(logits, action)
        return action, logp_a


class ActorCritic(nn.Module):
    """
    AC implementation that can be serialized by torchscript, and thus
    moved to the C++ code.
    """

    def __init__(self):
        super().__init__()
        self.pi = Actor()  # Policy
        self.v = Critic()  # Value function

    def forward(self, inputs):
        obs = inputs[:750]
        valid_action_mask = inputs[750:]
        action, logp_a = self.pi.forward(obs, valid_action_mask)
        v = self.v(obs)
        return torch.cat([action, logp_a, v])


def save_player(ac_model, total_iterations: int):
    # Overwrite the oldest AC if there are more than 5 in the directory
    ac_filenames = os.listdir(PLAYER_AC_LOC)
    if len(ac_filenames) > 4:
        ac_iters = [int(fn.split("_")[-1].split(".")[0]) for fn in ac_filenames]
        smallest_idx = ac_iters.index(min(ac_iters))
        ac_filename_to_delete = ac_filenames[smallest_idx]
        os.remove(PLAYER_AC_LOC + "\\" + ac_filename_to_delete)

    sm = torch.jit.script(ac_model)
    # print(sm.forward(torch.rand((2325))))
    player_ac = f"serialized_ac_{total_iterations}.pt"
    sm.save(PLAYER_AC_LOC + "\\" + player_ac)
    return player_ac


def save_opponent(ac_model, total_iterations: int):
    # Overwrite the oldest AC if there are more than 5 in the directory
    ac_filenames = os.listdir(OPPONENT_AC_LOC)
    if len(ac_filenames) > 4:
        ac_iters = [int(fn.split("_")[-1].split(".")[0]) for fn in ac_filenames]
        smallest_idx = ac_iters.index(min(ac_iters))
        ac_filename_to_delete = ac_filenames[smallest_idx]
        os.remove(OPPONENT_AC_LOC + "\\" + ac_filename_to_delete)

    sm = torch.jit.script(ac_model)
    opp_ac = f"serialized_ac_{total_iterations}.pt"
    sm.save(OPPONENT_AC_LOC + "\\" + opp_ac)
    return opp_ac


def save_actor_critic(ac_model):
    torch.save(ac_model, AC_LOC)


def loac_actor_critic():
    return torch.load(AC_LOC)


def get_actor_critic(load_from_disk=False):
    """Create a new Actor-Critic module or load one from disk."""
    if load_from_disk:
        if os.path.exists(AC_LOC):
            return loac_actor_critic()
    return ActorCritic()


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs, act, rew, val, logp, is_done, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam

        self.obs_buf = np.array(obs, dtype=np.float32)
        self.act_buf = np.array(act, dtype=np.float32)
        # self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.array(rew, dtype=np.float32)
        self.val_buf = np.array(val, dtype=np.float32)
        self.logp_buf = np.array(logp, dtype=np.float32)
        self.is_done = np.array(is_done, dtype=np.float32)

        self.adv_buf = np.zeros(len(self.obs_buf), dtype=np.float32)
        self.ret_buf = np.zeros(len(self.obs_buf), dtype=np.float32)

    def finish_path(self):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # TODO: fine-grained check of correctness.

        # Calculate this for every finished trajectory in the experience
        is_done_idx = np.where(self.is_done > 0)[0] + 1
        # Start the first trajectory at 0
        is_done_idx = np.insert(is_done_idx, 0, 0)
        for i in range(is_done_idx.size - 1):
            path_slice = slice(is_done_idx[i], is_done_idx[i + 1])
            rews = np.append(self.rew_buf[path_slice], 0)
            vals = np.append(self.val_buf[path_slice], 0)

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = (rews[:-1] + (self.gamma * vals[1:])) - vals[:-1]
            advantage = discount_cumsum(deltas, self.gamma * self.lam)
            self.adv_buf[path_slice] = advantage

            # the next line computes rewards-to-go, to be targets for the value function
            rewards_to_go = discount_cumsum(rews, self.gamma)[:-1]
            self.ret_buf[path_slice] = rewards_to_go

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        # the next two lines implement the advantage normalization trick
        adv_std = np.std(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class ActorDataset(Dataset):
    def __init__(self, data):
        self.obs = data["obs"]
        self.act = data["act"]
        self.adv = data["adv"]
        self.logp_old = data["logp"]
        assert len(self.obs) == len(self.act) == len(self.adv) == len(self.logp_old)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.adv[idx], self.logp_old[idx]


class CriticDataset(Dataset):
    def __init__(self, data):
        self.obs = data["obs"]
        self.ret = data["ret"]
        assert len(self.obs) == len(self.ret)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.ret[idx]


def compute_loss_pi(obs, act, adv, logp_old, ac):
    """Set up function for computing PPO policy loss"""
    # obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

    # Policy loss
    logp = ac.pi.forward_loss(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    # ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, cf=clipfrac)

    return loss_pi, pi_info


def compute_loss_v(obs, ret):
    """Set up function for computing value loss"""
    # obs, ret = data["obs"], data["ret"]
    pred_ret = ac.v(obs).squeeze()
    return ((pred_ret - ret) ** 2).mean()


def update_ppo(buffer, ac, batch_size, train_pi_iters, train_v_iters):
    # Build data loaders
    data = buffer.get()
    actor_ds = ActorDataset(data)
    critic_ds = CriticDataset(data)
    actor_dataloader = DataLoader(
        actor_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    critic_dataloader = DataLoader(
        critic_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        for obs, act, adv, logp_old in actor_dataloader:
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(obs, act, adv, logp_old, ac)
            print(pi_info)
            kl = pi_info["kl"]
            if kl > target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()
        if kl > target_kl:
            print("Early stopping at step %d due to reaching max kl." % i)
            break

    # Value function learning
    for i in range(train_v_iters):
        for obs, ret in critic_dataloader:
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(obs, ret)
            loss_v.backward()
            vf_optimizer.step()
        if i % 10 == 0:
            print(f"value loss: {loss_v.item()}")


def parse_env_experience(env_experience):
    env_experience = env_experience.decode("utf-8")
    env_experience = env_experience.split(",")
    observations = []
    observation = []
    actions = []
    rewards = []
    values = []
    action_logprobs = []
    is_done = []
    i = 0
    for val in env_experience:
        if val == "":
            break
        if i == 0:
            is_done.append(int(val))
        elif i == 1:
            actions.append(int(val))
        elif i == 2:
            rewards.append(int(val))
        elif i == 3:
            values.append(float(val))
        elif i == 4:
            action_logprobs.append(float(val))
        elif i == 755:
            i = -1
            observations.append(observation)
            observation = []
        elif i >= 5:
            observation.append(int(val))
        i += 1
    return observations, actions, rewards, values, action_logprobs, is_done


def play_games(ac_filename, n_games):
    # Environment process
    p = Popen(["build/Debug/train_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Separate inputs with newlines
    process_input = f"{ac_filename}\n{n_games}"
    output, err = p.communicate(process_input.encode("utf-8"))
    return output


if __name__ == "__main__":
    seed = 42
    gamma = 0.99
    lam = 0.97
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    batch_size = 512
    train_pi_iters = 80
    train_v_iters = 50
    target_kl = 0.4
    n_games = "100"
    total_iterations = 0

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create actor-critic module
    ac = get_actor_critic()
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    player_ac = save_player(ac, total_iterations)
    opp_ac = save_opponent(ac, total_iterations)

    # Main loop: collect experience in env and update AC
    while True:
        # Alwyas start playing games with the AC that has been trained the longest.
        # ac_filenames = os.listdir(PLAYER_AC_LOC)
        # ac_iters = [int(fn.split("_")[-1].split(".")[0]) for fn in ac_filenames]
        # largest_idx = ac_iters.index(max(ac_iters))
        # player_ac = ac_filenames[largest_idx]
        experience = play_games(player_ac, n_games)
        (
            observations,
            actions,
            rewards,
            values,
            action_logprobs,
            is_done,
        ) = parse_env_experience(experience)

        buffer = PPOBuffer(
            observations, actions, rewards, values, action_logprobs, is_done, gamma, lam
        )
        buffer.finish_path()
        update_ppo(buffer, ac, batch_size, train_pi_iters, train_v_iters)

        total_iterations += 1
        player_ac = save_player(ac, total_iterations)

        if total_iterations % 5 == 0 and total_iterations > 0:
            # If this player can reliably beat the best opponent, it becomes the new opponent.
            p = Popen(
                ["build/Debug/test_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE
            )
            # Separate inputs with newlines
            process_input = f"{player_ac}\n{opp_ac}"
            output, _ = p.communicate(process_input.encode("utf-8"))
            output = output.decode("utf-8")
            output = output.split(",")
            player_wins = output.count("P")
            print(f"Player wins: {player_wins}")
            if output.count("P") >= 18:
                opp_ac = save_opponent(ac, total_iterations)

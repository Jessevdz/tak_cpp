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
from itertools import accumulate
from subprocess import Popen, PIPE


SERIALIZED_AC_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\serialized_ac.pt"
AC_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\ac.pt"


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
            nn.Linear(750, 450),
            nn.ReLU(),
            nn.Linear(450, 250),
            nn.ReLU(),
            nn.Linear(250, 1),
        )

    def forward(self, obs):
        return self.v_net(obs)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(750, 850),
            nn.ReLU(),
            nn.Linear(850, 950),
            nn.ReLU(),
            nn.Linear(950, 1050),
            nn.ReLU(),
            nn.Linear(1050, 1250),
            nn.ReLU(),
            nn.Linear(1250, 1575),
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

    def forward(self, obs, valid_actions):
        """
        Produce action distributions for given observations
        """
        logits = self.policy_net(obs)
        masked_logits = torch.where(valid_actions > 0, logits, torch.tensor([-1e8]))
        masked_logits = masked_logits - masked_logits.logsumexp(dim=-1, keepdim=True)
        action = self.sample_action(masked_logits)
        logp_a = self.get_action_log_prob(masked_logits, action)
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
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def compute_loss_pi(data):
    """Set up function for computing PPO policy loss"""
    obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info


def compute_loss_v(data):
    """Set up function for computing value loss"""
    obs, ret = data["obs"], data["ret"]
    return ((ac.v(obs) - ret) ** 2).mean()


def serialize_actor_critic(ac_model):
    sm = torch.jit.script(ac_model)
    # print(sm.forward(torch.rand((2325))))
    sm.save(SERIALIZED_AC_LOC)


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


def update_ppo(buffer):
    data = buffer.get()

    pi_l_old, pi_info_old = compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        # kl = mpi_avg(pi_info["kl"])
        kl = pi_info["kl"]
        if kl > 1.5 * target_kl:
            print("Early stopping at step %d due to reaching max kl." % i)
            break
        loss_pi.backward()
        # mpi_avg_grads(ac.pi)  # average grads across MPI processes
        pi_optimizer.step()

    # Value function learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(data)
        loss_v.backward()
        mpi_avg_grads(ac.v)  # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    # kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
    # logger.store(
    #     LossPi=pi_l_old,
    #     LossV=v_l_old,
    #     KL=kl,
    #     Entropy=ent,
    #     ClipFrac=cf,
    #     DeltaLossPi=(loss_pi.item() - pi_l_old),
    #     DeltaLossV=(loss_v.item() - v_l_old),
    # )


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


def play_games(n_games):
    p = Popen(["build/Debug/tak_cpp.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(n_games)
    return output


if __name__ == "__main__":
    seed = 42
    epochs = 50
    gamma = 0.99
    lam = 0.97
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    target_kl = 0.01
    n_games = b"2"

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create actor-critic module
    ac = get_actor_critic()
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    save_actor_critic(ac)
    serialize_actor_critic(ac)

    # Main loop: collect experience in env and update AC
    for epoch in range(epochs):
        experience = play_games(n_games)
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
        update_ppo(buffer)
        serialize_actor_critic(ac)

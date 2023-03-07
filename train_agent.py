import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from train_utils import (
    discount_cumsum,
    AC_WEIGHTS_LOC,
    save_ac_weights,
    save_serialized_player,
    save_serialized_candidate,
    play_games,
    parse_env_experience,
    test_candidate,
)
import copy


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

        # Choose actions from masked logits
        masked_logits = torch.where(valid_actions > 0, logits, torch.tensor([-1e8]))
        action = self.sample_action(masked_logits)

        # But calculate logp from the unmasked logits
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


class CombinedAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
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
        self.actor = nn.Sequential(
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
        self.logsoftmax = nn.LogSoftmax()

    @torch.jit.export
    def sample_action(self, logits):
        # Logits to probabilities
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1, True)
        return action

    @torch.jit.export
    def get_action_log_prob(self, logits, act):
        # act = act.long().unsqueeze(-1)
        # act, log_pmf = torch.broadcast_tensors(act, logits)
        # act = act[..., :1]
        # return log_pmf.gather(-1, act).squeeze(-1)
        return logits.gather(1, act.unsqueeze(1).to(torch.int64)).squeeze(-1)

    def critic_loss(self, obs):
        return self.critic(obs)

    @torch.jit.export
    def actor_loss(self, obs, act):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.actor(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logp_a = self.get_action_log_prob(logits, act)
        return logp_a

    def forward(self, inputs):
        # Used in CPP code
        obs = inputs[:750]
        valid_action_mask = inputs[750:]
        # action, logp_a = self.actor_forward(obs, valid_action_mask)
        # v = self.critic_forward(obs)

        logits = self.actor(obs)
        # Normalize logits: https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py
        logits = logits - logits.logsumexp(dim=0)

        # Choose actions from masked logits
        masked_logits = torch.where(valid_action_mask > 0, logits, torch.tensor([-1e8]))
        action = self.sample_action(masked_logits)

        # But calculate logp from the unmasked logits
        # logp_a = self.get_action_log_prob(logits, action)
        logp_a = logits[action]
        v = self.critic(obs)
        return torch.cat([action, logp_a, v], dim=0)


def get_player(load_from_disk=False):
    """Create a new Actor-Critic module or load one from disk."""
    ac_module = ActorCritic()
    if load_from_disk:
        if os.path.exists(AC_WEIGHTS_LOC):
            # Load weights at the largest previously trained iteration
            weight_filename, iters = find_ac_weights(AC_WEIGHTS_LOC)
            ac_module.load_state_dict(
                torch.load(f"{AC_WEIGHTS_LOC}\\{weight_filename}")
            )
            return ac_module, iters
    return ac_module, 0


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

        assert (
            len(self.obs_buf)
            == len(self.act_buf)
            == len(self.rew_buf)
            == len(self.val_buf)
            == len(self.logp_buf)
            == len(self.is_done)
        )

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
        self.obs = data["obs"].to("cuda:0")
        self.act = data["act"].to("cuda:0")
        self.adv = data["adv"].to("cuda:0")
        self.logp_old = data["logp"].to("cuda:0")
        assert len(self.obs) == len(self.act) == len(self.adv) == len(self.logp_old)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx].to(torch.float32),
            self.act[idx].to(torch.float32),
            self.adv[idx].to(torch.float32),
            self.logp_old[idx].to(torch.float32),
        )


class CriticDataset(Dataset):
    def __init__(self, data):
        self.obs = data["obs"].to("cuda:0")
        self.ret = data["ret"].to("cuda:0")
        assert len(self.obs) == len(self.ret)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx].to(torch.float32), self.ret[idx].to(torch.float32)


def compute_loss_pi(obs, act, adv, logp_old, ac, clip_ratio=0.2):
    """Set up function for computing PPO policy loss"""
    # obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

    # Policy loss
    logp = ac.pi.forward_loss(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, cf=clipfrac)

    return loss_pi, pi_info


def compute_loss_v(ac, obs, ret):
    """Set up function for computing value loss"""
    # obs, ret = data["obs"], data["ret"]
    pred_ret = ac.v(obs).squeeze()
    return ((pred_ret - ret) ** 2).mean()


def update_ppo(buffer, ac_to_train, batch_size, train_pi_iters, train_v_iters):
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
        batch_step = 0
        for obs, act, adv, logp_old in actor_dataloader:
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(obs, act, adv, logp_old, ac_to_train)
            if i == 0 and batch_step == 0:
                print(pi_info)
                batch_step += 1
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
            loss_v = compute_loss_v(ac_to_train, obs, ret)
            loss_v.backward()
            vf_optimizer.step()
        # if i % 5 == 0:
        #     print(f"value loss: {loss_v.item()}")

    return ac_to_train


def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run(rank, size):
    seed = 42
    pi_lr = 3e-4
    vf_lr = 1e-3
    # 512 batch, 40 iters, 20 games
    batch_size = 512
    train_pi_iters = 40
    train_v_iters = 40
    target_kl = 5.0
    n_games = "20"

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create or load actor-critic module
    ac, total_iterations = get_player(load_from_disk=False)
    if rank == 0:
        # Serialize the actor-critic so that it can be loaded by environment processes
        save_serialized_player(ac.to("cpu"))
        # Save the candidate that new models need to try and beat
        save_serialized_candidate(ac.to("cpu"))

    dist.barrier()
    ac.to("cuda:0")
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    while True:
        experience = play_games(n_games)
        buffer = PPOBuffer(**parse_env_experience(experience))
        buffer.finish_path()
        # update_ppo(buffer, ac, batch_size, train_pi_iters, train_v_iters)
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
                # kl = pi_info["kl"]
                # if kl > target_kl:
                #     break
                loss_pi.backward()
                # Average gradients here
                average_gradients(ac.pi)
                pi_optimizer.step()
            if i % 5 == 0 and rank == 0:
                print(pi_info)
            # if kl > target_kl:
            #     print("Early stopping at step %d due to reaching max kl." % i)
            #     break

        dist.barrier()

        # Value function learning
        for i in range(train_v_iters):
            for obs, ret in critic_dataloader:
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(ac, obs, ret)
                loss_v.backward()
                # Average gradients here
                average_gradients(ac.v)
                vf_optimizer.step()
            if i % 5 == 0 and rank == 0:
                print(f"value loss: {loss_v.item()}")

        total_iterations += 1

        dist.barrier()

        if rank == 0:
            # Save a serialized version of the updated AC
            save_serialized_player(ac.to("cpu"))
            # Save the weights of the AC module if it can reliably beat its predecessor
            if total_iterations % 2 == 0 and total_iterations > 0:
                current_ac_wins = test_candidate()
                if current_ac_wins:
                    # The current AC becomes the new candidate to beat
                    save_serialized_candidate(ac)
                    # Save the weights of the AC module
                save_ac_weights(ac, total_iterations)
            ac.to("cuda:0")
        dist.barrier()


def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2222"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


# if __name__ == "__main__":
#     size = 2
#     processes = []
#     mp.set_start_method("spawn")
#     for rank in range(size):
#         p = mp.Process(target=init_process, args=(rank, size, run))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()


if __name__ == "__main__":
    seed = 42
    pi_lr = 3e-4
    vf_lr = 1e-3
    batch_size = 32
    train_pi_iters = 2
    train_v_iters = 2
    target_kl = 0.1
    n_games = "2"

    ac_to_train = CombinedAC()
    total_iterations = 0
    ac_to_train.to("cuda")
    pi_optimizer = Adam(ac_to_train.actor.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac_to_train.critic.parameters(), lr=vf_lr)

    ac_to_train.to("cpu")
    ac_to_train.eval()
    m_original = torch.jit.script(copy.deepcopy(ac_to_train))
    cur_module_name = f"module_{total_iterations}.pt"
    m_original.save(cur_module_name)

    # random_input = torch.rand((750))
    # random_mask = torch.where(torch.rand((1575)) > 0.7, 0, 1)
    # _inp = torch.cat([random_input, random_mask])
    # m_traced = torch.jit.trace(ac_to_train, _inp)

    # Main loop: collect experience in env and update AC
    while True:
        experience = play_games(cur_module_name, n_games)
        parsed_experience = parse_env_experience(experience)
        buffer = PPOBuffer(**parsed_experience)
        buffer.finish_path()

        # Train AC model
        data = buffer.get()
        actor_ds = ActorDataset(data)
        critic_ds = CriticDataset(data)
        actor_dataloader = DataLoader(
            actor_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        critic_dataloader = DataLoader(
            critic_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Train policy with multiple steps of gradient descent
        ac_to_train.to("cuda")
        ac_to_train.train()
        for i in range(train_pi_iters):
            batch_step = 0
            for obs, act, adv, logp_old in actor_dataloader:
                pi_optimizer.zero_grad()
                logp = ac_to_train.actor_loss(obs, act)
                ratio = torch.exp(logp - logp_old)
                clip_ratio = 0.2
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

                # Useful extra info
                approx_kl = (logp_old - logp).mean().item()
                clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, cf=clipfrac)
                if i == 0 and batch_step == 0:
                    print(ratio.std().item())
                    batch_step += 1
                kl = pi_info["kl"]
                if kl > target_kl:
                    break
                loss_pi.backward()
                pi_optimizer.step()
            if kl > target_kl:
                # print("Early stopping at step %d due to reaching max kl." % i)
                break

        # Value function learning
        for i in range(train_v_iters):
            for obs, ret in critic_dataloader:
                vf_optimizer.zero_grad()
                pred_ret = ac_to_train.critic_loss(obs).squeeze()
                loss_v = ((pred_ret - ret) ** 2).mean()
                loss_v.backward()
                vf_optimizer.step()

        torch.save(copy.deepcopy(ac_to_train.state_dict()), "tmp.pt")
        ac_to_train = CombinedAC()
        CombinedAC().load_state_dict(torch.load("tmp.pt"))
        ac_to_train.to("cuda")
        pi_optimizer = Adam(ac_to_train.actor.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac_to_train.critic.parameters(), lr=vf_lr)

        total_iterations += 1
        os.remove(cur_module_name)
        cur_module_name = f"module_{total_iterations}.pt"
        ac_to_train.to("cpu")
        ac_to_train.eval()
        m_current = torch.jit.script(copy.deepcopy(ac_to_train))
        m_current.save(cur_module_name)

        # ac_to_train.to("cpu")
        # ac_to_train.eval()
        # for obs, act, buf_logp in zip(buffer.obs_buf, buffer.act_buf, buffer.logp_buf):
        #     obs = torch.tensor(obs)
        #     act = torch.tensor(act)
        #     logp_original = round(
        #         m_original.actor_loss(obs.unsqueeze(0), act.unsqueeze(0)).item(), 3
        #     )
        #     logp_current = round(
        #         m_current.actor_loss(obs.unsqueeze(0), act.unsqueeze(0)).item(),
        #         3,
        #     )
        #     logp_ac = round(
        #         ac_to_train.actor_loss(obs.unsqueeze(0), act.unsqueeze(0)).item(), 3
        #     )
        #     logp_buffer = round(float(buf_logp), 3)
        #     pass

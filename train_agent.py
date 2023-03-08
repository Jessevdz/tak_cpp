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


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(750, 800)
        self.fc2 = nn.Linear(800, 850)
        self.fc3 = nn.Linear(850, 950)
        self.fc4 = nn.Linear(950, 1050)
        self.fc5 = nn.Linear(1050, 1150)
        self.fc6 = nn.Linear(1150, 1250)
        self.fc7 = nn.Linear(1250, 1350)
        self.fc8 = nn.Linear(1350, 1450)
        self.fc9 = nn.Linear(1450, 1575)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(750, 650)
        self.fc2 = nn.Linear(650, 550)
        self.fc3 = nn.Linear(550, 450)
        self.fc4 = nn.Linear(450, 350)
        self.fc5 = nn.Linear(350, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.critic = critic
        self.actor = actor

    @torch.jit.export
    def sample_action(self, logits):
        # Logits to probabilities
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1, True)
        return action

    @torch.jit.export
    def critic_loss(self, obs):
        return self.critic(obs)

    @torch.jit.export
    def actor_loss(self, obs, act):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.actor(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Get the log probabilities of the actions
        # obs and act should always be a batch here
        logp_a = logits.gather(1, act.unsqueeze(1).to(torch.int64)).squeeze(-1)
        return logp_a

    @torch.jit.ignore
    def to_torchscript(self):
        a = copy.deepcopy(self.actor).to("cpu").eval()
        c = copy.deepcopy(self.critic).to("cpu").eval()
        scripted_actor = torch.jit.script(a)
        scripted_critic = torch.jit.script(c)
        ac = ActorCritic(scripted_actor, scripted_critic).to("cpu").eval()
        scripted_ac = torch.jit.script(ac)
        return scripted_ac

    def forward(self, inputs, action=None):
        """Exported to torchscript module and used in the CPP code"""
        obs = inputs[:, 0:750]
        valid_action_mask = inputs[:, 750:2325]

        logits = self.actor(obs)
        # Normalize logits: https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py
        logits = logits - logits.logsumexp(dim=1, keepdim=True)

        v = None
        if action is None:
            # Choose actions from masked logits
            # masked_logits = torch.where(valid_action_mask > 0, logits, torch.tensor([-1e8]))
            masked_logits = torch.where(valid_action_mask > 0, logits, -1e8)
            # action = self.sample_action(masked_logits)
            probs = F.softmax(masked_logits, dim=1)
            action = torch.multinomial(probs, 1, True)
            v = self.critic(obs)
        else:
            action = action.long().unsqueeze(0)

        # But calculate logp from the unmasked logits
        # Here logits and action are not a batch, simply select the logp from logits.
        logp_a = logits.gather(1, action)

        action = torch.movedim(action, 0, 1)
        logp_a = torch.movedim(logp_a, 0, 1)
        if v is not None:
            return torch.cat([action, logp_a, v], dim=1)
        else:
            return torch.cat([action, logp_a], dim=1)


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
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


class ActorDataset(Dataset):
    def __init__(self, data):
        self.obs = data["obs"].to("cuda").to(torch.float32)
        self.act = data["act"].to("cuda").to(torch.float32)
        self.adv = data["adv"].to("cuda").to(torch.float32)
        self.logp_old = data["logp"].to("cuda").to(torch.float32)
        assert len(self.obs) == len(self.act) == len(self.adv) == len(self.logp_old)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx], self.act[idx], self.adv[idx], self.logp_old[idx])


class CriticDataset(Dataset):
    def __init__(self, data):
        self.obs = data["obs"].to("cuda").to(torch.float32)
        self.ret = data["ret"].to("cuda").to(torch.float32)
        assert len(self.obs) == len(self.ret)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.ret[idx]


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
    batch_size = 10
    train_pi_iters = 10
    train_v_iters = 10
    target_kl = 0.03
    n_games = "2"
    total_iterations = 0

    actor_gpu = Actor().to("cuda")
    critic_gpu = Critic().to("cuda")
    ac_gpu = ActorCritic(actor_gpu, critic_gpu).to("cuda")

    actor_cpu = Actor().to("cpu")
    for param in actor_cpu.parameters():
        param.requires_grad = False
    critic_cpu = Critic().to("cpu")
    for param in actor_cpu.parameters():
        param.requires_grad = False
    ac_cpu = ActorCritic(actor_cpu, critic_cpu).to("cpu")
    ac_cpu.eval()
    cpu_device = torch.device("cpu")

    random_input = torch.where(torch.rand((1, 750)) > 0.7, 0, 1)
    random_mask = torch.where(torch.rand((1, 1575)) > 0.7, 0, 1)
    inp = torch.cat([random_input, random_mask], dim=1)
    inp_cpu = copy.deepcopy(inp).to(torch.float32).cpu()
    inp_gpu = copy.deepcopy(inp).to(torch.float32).to("cuda")

    torch.save(ac_gpu.state_dict(), "tmp.pt")
    ac_cpu.load_state_dict(torch.load("tmp.pt", map_location=cpu_device))
    os.remove("tmp.pt")
    traced_ac_cpu = torch.jit.trace(ac_cpu, inp_cpu)
    # Save CPU traced module for CPP code
    torch.jit.save(traced_ac_cpu, "traced_ac_cpu.pt")

    # traced_ac_gpu = torch.jit.trace(ac_gpu, inp_gpu)
    # torch.jit.save(traced_ac_gpu, "traced_ac_gpu.pt")
    pi_optimizer = Adam(ac_gpu.actor.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac_gpu.critic.parameters(), lr=vf_lr)
    ac_gpu.train()

    # Main loop: collect experience in env and update AC
    while True:
        experience = play_games("traced_ac_cpu.pt", n_games)
        # os.remove("traced_ac_cpu.pt")
        parsed_experience = parse_env_experience(experience)
        buffer = PPOBuffer(**parsed_experience)
        buffer.finish_path()

        # Train AC model
        data = buffer.get()
        actor_ds = ActorDataset(data)
        critic_ds = CriticDataset(copy.deepcopy(data))
        actor_dataloader = DataLoader(
            actor_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        critic_dataloader = DataLoader(
            critic_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Train policy with multiple steps of gradient descent
        first = True
        for i in range(train_pi_iters):
            for obs, act, adv, logp_old in actor_dataloader:
                pi_optimizer.zero_grad()
                out = ac_gpu(obs, act)
                logp = out[:, 1]
                ratio = torch.exp(logp - logp_old)
                clip_ratio = 0.2
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

                # Useful extra info
                approx_kl = (logp_old - logp).mean().item()
                clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, cf=clipfrac)

                if first and i == 0:
                    print(ratio.std().item())
                    print(f"kl: {pi_info['kl']}")
                    first = False

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
                pred_ret = ac_gpu.critic(obs).squeeze()
                loss_v = ((pred_ret - ret) ** 2).mean()
                loss_v.backward()
                vf_optimizer.step()

        total_iterations += 1

        # Save CPU traced module for CPP code
        torch.save(ac_gpu.state_dict(), "tmp.pt")
        actor_cpu = Actor().to("cpu")
        for param in actor_cpu.parameters():
            param.requires_grad = False
        critic_cpu = Critic().to("cpu")
        for param in actor_cpu.parameters():
            param.requires_grad = False
        ac_cpu = ActorCritic(actor_cpu, critic_cpu).to("cpu")
        ac_cpu.eval()
        ac_cpu.load_state_dict(torch.load("tmp.pt", map_location=cpu_device))
        os.remove("tmp.pt")

        inp_cpu = copy.deepcopy(inp).to(torch.float32).cpu()
        traced_ac_cpu = torch.jit.trace(ac_cpu, inp_cpu)
        # Save CPU traced module for CPP code
        torch.jit.save(traced_ac_cpu, "traced_ac_cpu.pt")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from train_utils import (
    discount_cumsum,
    AC_WEIGHTS_LOC,
    SERIALIZED_PLAYER_LOC,
    ACTOR_WEIGHTS_LOC,
    play_games,
    parse_env_experience,
    test_candidate,
)
from scipy.stats import entropy
import shutil


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

    def get_logits(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        logits = self.fc9(x)
        return logits

    @torch.jit.ignore
    def forward_loss(self, obs, act):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.get_logits(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Get the log probabilities of the actions
        # obs and act should always be a batch here
        logp_a = logits.gather(1, act.unsqueeze(1).to(torch.int64)).squeeze(-1)
        return logp_a

    def forward(self, inputs):
        obs = inputs[:, 0:750]
        valid_action_mask = inputs[:, 750:2325]
        logits = self.get_logits(obs)
        # Normalize logits: https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py
        logits = logits - logits.logsumexp(dim=1, keepdim=True)
        masked_logits = torch.where(valid_action_mask > 0, logits, -1e8)
        probs = F.softmax(masked_logits, dim=1)
        action = torch.multinomial(probs, 1, True)
        logp_a = logits.gather(1, action)
        action = torch.movedim(action, 0, 1)
        logp_a = torch.movedim(logp_a, 0, 1)
        return torch.cat([action, logp_a], dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc10 = nn.Linear(750, 650)
        self.fc20 = nn.Linear(650, 550)
        self.fc30 = nn.Linear(550, 450)
        self.fc40 = nn.Linear(450, 450)
        self.fc50 = nn.Linear(450, 450)
        self.fc60 = nn.Linear(450, 350)
        self.fc70 = nn.Linear(350, 1)

    def forward(self, inputs):
        obs = inputs[:, 0:750]
        x = torch.tanh(self.fc10(obs))
        x = torch.tanh(self.fc20(x))
        x = torch.tanh(self.fc30(x))
        x = torch.tanh(self.fc40(x))
        x = torch.tanh(self.fc50(x))
        x = torch.tanh(self.fc60(x))
        x = self.fc70(x)
        return x


def get_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    ent = entropy(counts, base=base)
    return ent


def save_and_serialize_model(module, nr_iterations: int, actor=True):
    # Overwrite the oldest AC if there are more than 5 in the directory
    if len(os.listdir(AC_WEIGHTS_LOC)) > 10:
        # Delete the oldest weights
        filenames = os.listdir(AC_WEIGHTS_LOC)
        iters_trained = [int(fn.split("_")[0]) for fn in filenames]
        iters = min(iters_trained)
        os.remove(f"{AC_WEIGHTS_LOC}\\{iters}_it_actor_weights.pt")
        os.remove(f"{AC_WEIGHTS_LOC}\\{iters}_it_critic_weights.pt")

    # Safely store GPU model weights in CPU model
    actor_weights_loc = f"{AC_WEIGHTS_LOC}\\{nr_iterations}_it_actor_weights.pt"
    critic_weights_loc = f"{AC_WEIGHTS_LOC}\\{nr_iterations}_it_critic_weights.pt"

    if actor:
        torch.save(
            module.state_dict(),
            actor_weights_loc,
        )
        actor_cpu.load_state_dict(
            torch.load(actor_weights_loc, map_location=cpu_device)
        )
        traced_actor_cpu = torch.jit.trace(actor_cpu, actor_input, check_trace=False)
        torch.jit.save(traced_actor_cpu, f"{SERIALIZED_PLAYER_LOC}\\actor.pt")
    else:
        torch.save(
            module.state_dict(),
            critic_weights_loc,
        )
        critic_cpu.load_state_dict(
            torch.load(critic_weights_loc, map_location=cpu_device)
        )
        traced_critic_cpu = torch.jit.trace(critic_cpu, critic_input, check_trace=False)
        torch.jit.save(traced_critic_cpu, f"{SERIALIZED_PLAYER_LOC}\\critic.pt")


def create_opponent_player():
    # Copy the current actor and critic - these are the opponents that need to be beaten.
    shutil.copyfile(
        f"{SERIALIZED_PLAYER_LOC}\\actor.pt",
        f"{SERIALIZED_PLAYER_LOC}\\actor_opponent.pt",
    )


if __name__ == "__main__":
    seed = 42
    pi_lr = 1e-4
    vf_lr = 1e-4
    gamma = 0.99
    lam = 0.95
    batch_size_pi = 128
    batch_size_v = 128
    train_pi_iters = 5
    train_v_iters = 10
    target_kl = 0.03
    n_games = "10"
    train_iterations = 0

    # Set up Actor and Critic for inference on GPU
    actor_gpu = Actor().to("cuda")
    critic_gpu = Critic().to("cuda")
    # CPU actor and critics - to be used by CPP code
    actor_cpu = Actor().to("cpu")
    critic_cpu = Critic().to("cpu")

    # Inputs for tracing CPU actor and critic
    cpu_device = torch.device("cpu")
    random_input = torch.where(torch.rand((1, 750)) > 0.7, 0, 1)
    random_mask = torch.where(torch.rand((1, 1575)) > 0.7, 0, 1)
    actor_input = torch.cat([random_input, random_mask], dim=1).to(torch.float32).cpu()
    critic_input = random_input.to(torch.float32).cpu()

    # Restart from previous training runs if they are on disk
    filenames = os.listdir(AC_WEIGHTS_LOC)
    if filenames:
        iters_trained = [int(fn.split("_")[0]) for fn in filenames]
        iters = max(iters_trained)
        actor_gpu.load_state_dict(
            torch.load(f"{AC_WEIGHTS_LOC}\\{iters}_it_actor_weights.pt")
        )
        critic_gpu.load_state_dict(
            torch.load(f"{AC_WEIGHTS_LOC}\\{iters}_it_critic_weights.pt")
        )
        train_iterations = iters

    save_and_serialize_model(actor_gpu, train_iterations)
    save_and_serialize_model(critic_gpu, train_iterations, False)
    create_opponent_player()

    # Set up optimizers
    pi_optimizer = Adam(actor_gpu.parameters(), lr=pi_lr)
    vf_optimizer = Adam(critic_gpu.parameters(), lr=vf_lr)

    # Main loop: collect experience in env and update AC
    while True:
        actor_gpu.train()
        critic_gpu.train()

        experience = play_games("player", n_games)
        parsed_experience = parse_env_experience(experience)
        parsed_experience = {key: np.array(l) for key, l in parsed_experience.items()}

        # Calculate advantage and rewards to go
        adv_buf = np.zeros(len(parsed_experience["is_done"]), dtype=np.float32)
        ret_buf = np.zeros(len(parsed_experience["is_done"]), dtype=np.float32)
        # +1 because otherwise the actual reward is not included in the slice below
        is_done_idx = np.where(parsed_experience["is_done"] > 0)[0] + 1
        # Start the first trajectory at 0
        is_done_idx = np.insert(is_done_idx, 0, 0)
        for i in range(is_done_idx.size - 1):
            path_slice = slice(is_done_idx[i], is_done_idx[i + 1])
            rews = np.append(parsed_experience["rew"][path_slice], 0)
            vals = np.append(parsed_experience["val"][path_slice], 0)

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = (rews[:-1] + (gamma * vals[1:])) - vals[:-1]
            advantage = discount_cumsum(deltas, gamma * lam)
            adv_buf[path_slice] = advantage

            # the next line computes rewards-to-go, to be targets for the value function
            rewards_to_go = discount_cumsum(rews, gamma)[:-1]
            ret_buf[path_slice] = rewards_to_go

        adv_std = np.std(adv_buf)
        adv_mean = np.mean(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std

        # How many batches to train over
        nr_observations = len(parsed_experience["obs"])
        print(nr_observations)
        pi_batches = int(nr_observations / batch_size_pi)
        v_batches = int(nr_observations / batch_size_v)

        # Train policy with multiple steps of gradient descent
        new_experience = True
        for train_pi_iter in range(train_pi_iters):
            batch_step = 0
            # for obs, act, adv, logp_old in actor_dataloader:
            for i in range(pi_batches):
                # Get data
                range_start = i * batch_size_pi
                if i == pi_batches - 1:
                    range_end = nr_observations + 1
                else:
                    range_end = (i + 1) * batch_size_pi
                obs = parsed_experience["obs"][range_start:range_end]
                act = parsed_experience["act"][range_start:range_end]
                adv = adv_buf[range_start:range_end]
                logp_old = parsed_experience["logp"][range_start:range_end]

                obs = torch.from_numpy(obs).to(torch.float32).to("cuda")
                act = torch.from_numpy(act).to(torch.float32).to("cuda")
                adv = torch.from_numpy(adv).to(torch.float32).to("cuda")
                logp_old = torch.from_numpy(logp_old).to(torch.float32).to("cuda")

                pi_optimizer.zero_grad()
                # out = ac_gpu(obs, act)
                # logp = out[:, 1]
                logp = actor_gpu.forward_loss(obs, act)
                ratio = torch.exp(logp - logp_old)
                clip_ratio = 0.2
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

                # Useful extra info
                approx_kl = (logp_old - logp).mean().item()
                # clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                # pi_info = dict(kl=approx_kl, cf=clipfrac)

                if new_experience and i == 0:
                    # Check that the predictions on python side are the same
                    # as those on CPP side.
                    # print(ratio.std().item())
                    assert ratio.std().item() < 1e-4
                    new_experience = False
                    # print(f"logp entropy: {get_entropy(parsed_experience['logp'])}")
                    # print(f"val entropy: {get_entropy(parsed_experience['val'])}")

                if approx_kl > target_kl:
                    break
                loss_pi.backward()
                pi_optimizer.step()
                batch_step += 1
            if approx_kl > target_kl:
                print(
                    f"Early stopping at epoch {train_pi_iter} batch step {batch_step} due to reaching max kl: {approx_kl}"
                )
                break

        # Value function learning
        print("Value loss:")
        for train_v_iter in range(train_v_iters):
            train_steps = 0
            epoch_loss = 0
            # for obs, ret in critic_dataloader:
            for i in range(v_batches):
                # Get data
                range_start = i * batch_size_v
                if i == v_batches - 1:
                    range_end = nr_observations + 1
                else:
                    range_end = (i + 1) * batch_size_v
                obs = parsed_experience["obs"][range_start:range_end]
                ret = ret_buf[range_start:range_end]
                obs = torch.from_numpy(obs).to(torch.float32).to("cuda")
                ret = torch.from_numpy(ret).to(torch.float32).to("cuda")

                vf_optimizer.zero_grad()
                pred_ret = critic_gpu(obs).squeeze()
                loss_v = ((pred_ret - ret) ** 2).mean()
                epoch_loss += loss_v.item()
                train_steps += 1
                loss_v.backward()
                vf_optimizer.step()
            print(epoch_loss / train_steps)

        train_iterations += 1
        save_and_serialize_model(actor_gpu, train_iterations)
        save_and_serialize_model(critic_gpu, train_iterations, False)

        if train_iterations % 5 == 0 and train_iterations > 0:
            player_wins = test_candidate()
            if player_wins:
                # Current player becomes new opponent to beat
                create_opponent_player()
                # Copy the current weights to a backup folder
                loc = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\backups"
                shutil.copyfile(
                    f"{AC_WEIGHTS_LOC}\\{train_iterations}_it_critic_weights.pt",
                    f"{loc}\\{train_iterations}_it_critic_weights.pt",
                )
                shutil.copyfile(
                    f"{AC_WEIGHTS_LOC}\\{train_iterations}_it_actor_weights.pt",
                    f"{loc}\\{train_iterations}_it_actor_weights.pt",
                )

"""
PPO implementation from Gym: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
"""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import time
import torch.nn.functional as F


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


class Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, 850),
            nn.ReLU(),
            nn.Linear(850, 950),
            nn.ReLU(),
            nn.Linear(950, 1050),
            nn.ReLU(),
            nn.Linear(1050, 1150),
            nn.ReLU(),
            nn.Linear(1150, action_dim),
        )

    def _distribution(self, obs, valid_action_mask):
        logits = self.policy_net(obs)
        masked_logits = torch.where(
            valid_action_mask > 0, logits, torch.tensor([-100000000.0])
        )
        return Categorical(logits=masked_logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, valid_action_mask, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, valid_action_mask)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


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


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        # Policy
        self.pi = Actor(observation_dim, action_dim)
        # Value function
        self.v = Critic()

    def step(self, obs, valid_action_mask):
        with torch.no_grad():
            pi = self.pi._distribution(obs, valid_action_mask)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ActorTS(nn.Module):
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


class ActorCriticTS(nn.Module):
    """
    AC implementation that can be serialized by torchscript, and thus
    moved to the C++ code.
    """

    def __init__(self):
        super().__init__()
        self.pi = ActorTS()  # Policy
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

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
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

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
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


def ppo(
    env_fn,
    actor_critic=ActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
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

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    seed = 42
    steps_per_epoch = 4000
    epochs = 50
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.01
    logger_kwargs = dict()
    save_freq = 10

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
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

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

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

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            # logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs - 1):
        #     logger.save_state({"env": env}, None)

        # Perform PPO update!
        update()

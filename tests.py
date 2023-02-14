from ppo import *
import torch


def test_actor_critic():
    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1275)) > 0.7, 1, 0)
    ac = MLPActorCritic(observation_dim=750, action_dim=1275)
    a, v, logp_a = ac.step(random_input, random_mask)
    print(a)
    print(v)
    print(logp_a)


if __name__ == "__main__":
    test_actor_critic()

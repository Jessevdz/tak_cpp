from ppo import *
import torch


def test_actor_critic():
    torch.manual_seed(42)
    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1275)) > 0.7, 0, 1)
    ac = ActorCritic(observation_dim=750, action_dim=1275)
    a, v, logp_a = ac.step(random_input, random_mask)
    print(a)
    print(v)
    print(logp_a)
    # Save network weights and random vectors for comparison on the CPP side.
    ac.save_actor()
    ac.save_critic()
    torch.save(
        random_input, "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\random_input.pt"
    )
    torch.save(random_mask, "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\random_mask.pt")


def test_convert_to_torchscript():
    class Critic(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self.v_net = nn.Sequential(
                nn.Linear(obs_dim, 450),
                nn.ReLU(),
                nn.Linear(450, 250),
                nn.ReLU(),
                nn.Linear(250, 1),
            )

        def forward(self, obs):
            return torch.squeeze(
                self.v_net(obs), -1
            )  # Critical to ensure v has right shape.

    critic = Critic(750)
    sm = torch.jit.script(critic)
    sm.save("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt")


if __name__ == "__main__":
    # test_actor_critic()
    test_convert_to_torchscript()

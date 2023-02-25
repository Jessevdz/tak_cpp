from ppo import *


if __name__ == "__main__":
    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1575)) > 0.7, 0, 1)
    ac = ActorCritic()
    a, logp_1 = ac.pi.forward(random_input, random_mask)
    logp_2 = ac.pi.forward_loss(random_input, a)
    pass

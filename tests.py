from ppo import *
import torch


def compare_AC_outputs():
    """
    Torchscript cannot handle the Categorical distribution, which means
    we need to reimplement it manually.

    If you switch around the order of the two implementations,
    the outputs should remain the same.
    """
    torch.manual_seed(42)

    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1275)) > 0.7, 0, 1)

    ac_1 = ActorCritic(observation_dim=750, action_dim=1275)
    a, v, logp_a = ac_1.step(random_input, random_mask)
    print(a)
    print(v)
    print(logp_a)

    ac_2 = ActorCriticTS()
    a, v, logp_a = ac_2.forward(torch.cat([random_input, random_mask]))
    print(a)
    print(v)
    print(logp_a)


def test_convert_to_torchscript():
    ac = ActorCriticTS()
    sm = torch.jit.script(ac)
    print(sm.forward(torch.rand((2325))))
    sm.save("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt")


if __name__ == "__main__":
    # compare_AC_outputs()
    test_convert_to_torchscript()

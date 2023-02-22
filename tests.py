from ppo import *
import torch
import pandas as pd


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


def read_exp_csv():
    df = pd.read_csv("data/experience_1000_games.csv", usecols=[1])
    vc = df.value_counts()
    played_moves = [l[0] for l in vc.index.to_list()]
    all_moves = list(range(1575))
    for m in all_moves:
        if m not in played_moves:
            print(m)
    pass


def parse_experience(str_output):
    str_output = str_output.split(",")
    observations = []
    observation = []
    actions = []
    rewards = []
    values = []
    action_logprobs = []
    is_done = []
    i = 0
    for val in str_output:
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
    pass


if __name__ == "__main__":
    # compare_AC_outputs()
    # test_convert_to_torchscript()
    # read_exp_csv()

    from subprocess import Popen, PIPE

    p = Popen(["build/Debug/tak_cpp.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"2")
    str_output = output.decode("utf-8")
    parse_experience(str_output)
    pass

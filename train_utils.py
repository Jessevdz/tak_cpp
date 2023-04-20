AC_WEIGHTS_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\ac_weights"
ACTOR_WEIGHTS_LOC = f"{AC_WEIGHTS_LOC}\\actor_weights.pt"
CRITIC_WEIGHTS_LOC = f"{AC_WEIGHTS_LOC}\\critic_weights.pt"
# Current player gathering experience in the environments
SERIALIZED_PLAYER_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\traced"


import torch
import scipy.signal
import os
from subprocess import Popen, PIPE


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


def find_ac_weights(loc: str, newest=True):
    """
    Find the weights in loc that have trained for the most iterations
    Return both the weight filename and the nr of iterations
    """
    filenames = os.listdir(loc)
    iters_trained = [int(fn.split("_")[0]) for fn in filenames]
    if newest:
        idx = iters_trained.index(max(iters_trained))
        iters = max(iters_trained)
    else:  # Oldest
        idx = iters_trained.index(min(iters_trained))
        iters = min(iters_trained)
    return filenames[idx], iters


def save_ac_weights(ac_module, nr_iterations: int):
    # Overwrite the oldest AC if there are more than 5 in the directory
    # if len(os.listdir(AC_WEIGHTS_LOC)) > 4:
    #     # Delete the oldest weights
    #     weights_loc, _ = find_ac_weights(AC_WEIGHTS_LOC, newest=False)
    #     os.remove(f"{AC_WEIGHTS_LOC}\\{weights_loc}")
    torch.save(
        ac_module.state_dict(), f"{AC_WEIGHTS_LOC}\\{nr_iterations}_it_weights.pt"
    )


def save_actor_weights(module, nr_iterations: int):
    torch.save(
        module.state_dict(), f"{AC_WEIGHTS_LOC}\\{nr_iterations}_it_actor_weights.pt"
    )


def play_games(path, n_games):
    """Play a number of games with an agent and return the experience."""
    # Environment process
    p = Popen(
        ["build/Debug/train_env.exe"],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    # Separate inputs with newlines
    process_input = f"{n_games}\n"
    output, err = p.communicate(process_input.encode("utf-8"))
    p.kill()
    return output


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
            continue
        elif i == 0:
            is_done.append(int(val))
        elif i == 1:
            actions.append(int(val))
        elif i == 2:
            rewards.append(int(val))
        elif i == 3:
            values.append(float(val))
        elif i == 4:
            action_logprobs.append(float(val))
        elif i == 754:
            i = -1
            observation.append(int(val))
            observations.append(observation)
            observation = []
        elif i >= 5:
            observation.append(int(val))
        i += 1
    assert (
        len(values)
        == len(rewards)
        == len(observations)
        == len(is_done)
        == len(actions)
        == len(action_logprobs)
        == (len(env_experience) - 1) / 755
    )
    for obs in observations:
        assert min(obs) >= 0
        assert max(obs) <= 1
        assert len(obs) == 750
    return {
        "obs": observations,
        "act": actions,
        "rew": rewards,
        "val": values,
        "logp": action_logprobs,
        "is_done": is_done,
    }


def test_candidate():
    p = Popen(["build/Debug/test_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Separate inputs with newlines
    # process_input = f"{SERIALIZED_PLAYER_LOC}\n{CANDIDATE_LOC}"
    # output, _ = p.communicate(process_input.encode("utf-8"))
    output, _ = p.communicate()
    output = output.decode("utf-8")
    output = output.split(",")
    player_wins = output.count("P")
    print(f"Player wins: {player_wins}")
    if output.count("P") >= 18:
        return True
    else:
        return False

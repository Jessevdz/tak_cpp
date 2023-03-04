AC_WEIGHTS_LOC = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\players"
# The current candidate that new models need to try and beat
CANDIDATE_LOC = (
    "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\serialized\\candidate_player.pt"
)
# Current player gathering experience in the environments
SERIALIZED_PLAYER_LOC = (
    "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\serialized\\env_player.pt"
)


import torch
import scipy.signal
import os
from subprocess import Popen, PIPE
import copy
import io


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
    if len(os.listdir(AC_WEIGHTS_LOC)) > 4:
        # Delete the oldest weights
        weights_loc, _ = find_ac_weights(AC_WEIGHTS_LOC, newest=False)
        os.remove(f"{AC_WEIGHTS_LOC}\\{weights_loc}")
    torch.save(
        ac_module.state_dict(), f"{AC_WEIGHTS_LOC}\\{nr_iterations}_it_weights.pt"
    )


def save_serialized(ac, loc):
    module_to_save = copy.deepcopy(ac)
    module_to_save.to("cpu")
    # module_to_save.eval()
    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1575)) > 0.7, 0, 1)
    example_input = torch.cat([random_input, random_mask])
    # m = torch.jit.script(ac)
    m = torch.jit.trace(ac, example_input)
    # Save to file
    import io

    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    # m.save(loc)
    return buffer


def save_serialized_player(ac):
    save_serialized(ac, SERIALIZED_PLAYER_LOC)


def save_serialized_candidate(ac):
    save_serialized(ac, CANDIDATE_LOC)


def play_games(n_games):
    """Play a number of games with an agent and return the experience."""
    # Environment process
    p = Popen(["build/Debug/train_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Separate inputs with newlines
    process_input = f"{SERIALIZED_PLAYER_LOC}\n{n_games}"
    output, err = p.communicate(process_input.encode("utf-8"))
    p.kill()
    return output


def play_games_2(ac):
    # Environment process
    buf = io.BytesIO()
    buf_size = buf.write(torch.jit.script(ac).save_to_buffer())
    p = Popen(["build/Debug/train_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    buf.seek(0)
    output, err = p.communicate(buf.read(10))
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
    process_input = f"{SERIALIZED_PLAYER_LOC}\n{CANDIDATE_LOC}"
    output, _ = p.communicate(process_input.encode("utf-8"))
    output = output.decode("utf-8")
    output = output.split(",")
    player_wins = output.count("P")
    print(f"Player wins: {player_wins}")
    if output.count("P") >= 18:
        return True
    else:
        return False

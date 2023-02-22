#include "board.h"
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <memory>
#include <chrono>

struct Experience
{
    vector<int> observation;
    int action;
    float reward;
    float value;
    float action_logprob;
    int is_done;
};

struct ExperienceBuffer
{
    vector<vector<int>> observations;
    vector<int> actions;
    vector<float> rewards;
    vector<float> values;
    vector<float> action_logprobs;
    vector<int> is_done;
    int experience_len = 0;

    // Append one timestep of agent-environment interaction to the buffer.
    void add_experience(Experience &exp)
    {
        observations.push_back(exp.observation);
        actions.push_back(exp.action);
        rewards.push_back(exp.reward);
        values.push_back(exp.value);
        action_logprobs.push_back(exp.action_logprob);
        is_done.push_back(exp.is_done);
        experience_len++;
    }
};

/******************************************************
A Tak environment has two players, which are controlled
by the same policy, and a board.
*******************************************************/
class TakEnv
{
private:
    ExperienceBuffer white_player_experience;
    ExperienceBuffer black_player_experience;
    void write_player_experience(string, ExperienceBuffer &);

public:
    TakEnv();
    torch::jit::script::Module player;
    Board board;
    void reset();
    bool step();
    void write_experience_to_cout();
};

/******************************************************
Initialize players with trained model. Reset the board.
*******************************************************/
TakEnv::TakEnv()
{
    player = torch::jit::load("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\serialized_ac.pt");
    ExperienceBuffer white_player_experience = ExperienceBuffer();
    ExperienceBuffer black_player_experience = ExperienceBuffer();
    player.eval();
    reset();
}

/******************************************************
Reset the board state
*******************************************************/
void TakEnv::reset()
{
    board.reset_board();
}

/******************************************************
Take an observation from the board.
Have the player module pick an action.
Perform the action on the board.
Save all necessary state.
*******************************************************/
bool TakEnv::step()
{
    // Get necessary state from the board
    vector<int> obs = board.get_board_state();
    vector<int> moves_mask = board.get_valid_moves_mask();
    // Create Tensor input
    torch::Tensor obs_tensor = torch::from_blob(obs.data(), obs.size(), torch::TensorOptions().device(torch::kCPU));
    torch::Tensor moves_tensor = torch::from_blob(moves_mask.data(), moves_mask.size(), torch::TensorOptions().device(torch::kCPU));
    torch::Tensor arr[2] = {obs_tensor, moves_tensor};
    torch::Tensor input_tensor = torch::cat(arr);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    // Pick a move
    at::Tensor output = player.forward(inputs).toTensor();
    // Output contains [action, logp_a, v]
    int action = output[0].item<int>();
    float logp_a = output[1].item<float>();
    float value = output[2].item<float>();
    // Get the player executing the move.
    char active_player = board.get_active_player();
    // Execute the chosen move on the board.
    WinConditions win_conditions = board.take_action(action);
    bool game_ends = win_conditions.game_ends;
    // Append the reward.
    // If the active player made a move that had the opponent win: append -1 reward.
    // If the active player made a move that resulted in a win: append +1 reward.
    int reward = 0;
    int is_done = 0;
    if (game_ends)
    {
        char winner = win_conditions.game_ends;
        if (winner == 'T') // A tie in a flat win.
        {
            reward = 0;
        }
        else if (winner == active_player)
        {
            reward = 1;
        }
        else
        {
            reward = -1;
        }
    }
    Experience exp = {obs, action, reward, value, logp_a, is_done};
    if (active_player == 'W')
    {
        white_player_experience.add_experience(exp);
    }
    else
    {
        black_player_experience.add_experience(exp);
    }
    if (game_ends)
    {
        // Make sure is_done is set correctly for both players
        white_player_experience.is_done.back() = 1;
        black_player_experience.is_done.back() = 1;
    }
    return game_ends;
}

/******************************************************
Write a single player's experience to disk.
*******************************************************/
void TakEnv::write_player_experience(string ss, ExperienceBuffer &player_exp)
{
    char delimiter = ',';
    for (int i = 0; i < player_exp.experience_len; i++)
    {
        int action = player_exp.actions[i];
        int reward = player_exp.rewards[i];
        float value = player_exp.values[i];
        float action_logprob = player_exp.action_logprobs[i];
        int is_done = player_exp.is_done[i];
        vector<int> obs = player_exp.observations[i];
        ss += std::to_string(is_done) += delimiter;
        ss += std::to_string(action) += delimiter;
        ss += std::to_string(reward) += delimiter;
        ss += std::to_string(value) += delimiter;
        ss += std::to_string(action_logprob) += delimiter;
        ss += std::to_string(is_done) += delimiter;
        for (int i : obs)
        {
            ss += std::to_string(i) += delimiter;
        }
        std::cout << ss;
        ss.clear();
    }
}

/******************************************************
Write the combined white and black player experience to disk.
*******************************************************/
void TakEnv::write_experience_to_cout()
{
    string ss;
    ss.reserve(2000);
    write_player_experience(ss, white_player_experience);
    write_player_experience(ss, black_player_experience);
}

int main()
{
    int games_to_play;
    cin >> games_to_play;
    TakEnv env = TakEnv();
    for (int i = 0; i < games_to_play; i++)
    {
        bool game_ends = false;
        while (!game_ends)
        {
            game_ends = env.step();
        }
        env.reset();
    }
    env.write_experience_to_cout();
}
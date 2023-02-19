#include "board.h"
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <memory>

struct GameState
{
    // The NEW state of the board due to the player taking an action
    vector<int> observation;
    // Contains reward: The reward as a result of taking the action.
    // Contains termination info: whether the game is done.
    // If game is done, we need to call reset()
    WinConditions win_conditions;
};

/******************************************************
A Tak environment has two players, which are controlled
by the same policy, and a board.
*******************************************************/
class TakEnv
{
private:
public:
    TakEnv();
    torch::jit::script::Module player;
    Board board;
    void reset();
    void step();
};

/******************************************************
Initialize players with trained model.
Reset the board, and return the initial observation.
*******************************************************/
TakEnv::TakEnv()
{
    player = torch::jit::load("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt");
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
void TakEnv::step()
{
    vector<int> obs = board.get_board_state();
    torch::Tensor obs_tensor = torch::from_blob(obs.data(), obs.size(), torch::TensorOptions().device(torch::kCPU));
    std::cout << obs_tensor << std::endl;
    // at::Tensor ac_output = player.forward(inputs).toTensor();
}

/******************************************************
Take an action on the board, and return
an observation, reward, and termination info.
******************************************************/
// GameState TakEnv::step(int action)
// {
//     WinConditions win_conditions = board.take_action(action);
// }

int load_and_run_ts_module()
{
    // Test torchscript
    torch::jit::script::Module module;
    module = torch::jit::load("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt");
    torch::NoGradGuard no_grad;
    module.eval();

    torch::Tensor random_obs = torch::rand({750});
    torch::Tensor random_mask = torch::where(torch::rand({1275}) > 0.7, 0, 1);
    torch::Tensor arr[2] = {random_obs, random_mask};
    torch::Tensor input_tensor = torch::cat(arr);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute the model and turn its output into a tensor.
    // It's imperative the output returns a tensor
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output << '\n';
    return 0;
}

int main()
{
    TakEnv env = TakEnv();
    env.step();
}
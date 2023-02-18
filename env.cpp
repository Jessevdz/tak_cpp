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
    Board board = Board();
    void reset();
    GameState step(int);
};

/******************************************************
Reset the board state, return the initial observation.
*******************************************************/
void TakEnv::reset()
{
    board = Board();
}

/******************************************************
Take an action on the board, and return
an observation, reward, and termination info.
******************************************************/
// GameState TakEnv::step(int action)
// {
//     WinConditions win_conditions = board.take_action(action);
// }

int main()
{
    torch::NoGradGuard no_grad;
    // Test torchscript
    torch::jit::script::Module module;
    module = torch::jit::load("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt");
    torch::Tensor random_obs = torch::rand({750});
    torch::Tensor random_mask = torch::where(torch::rand({1275}) > 0.7, 0, 1);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(random_obs);
    inputs.push_back(random_mask);
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output << '\n';
    return 0;
}
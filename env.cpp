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
Actor Critic implementations
*******************************************************/
struct ActorOutput
{
    int a;        // Action integer
    float logp_a; // Log probability of the action
};

struct Actor : torch::nn::Module
{
    Actor()
    {
        fc1 = register_module("fc1", torch::nn::Linear(750, 850));
        fc2 = register_module("fc2", torch::nn::Linear(850, 950));
        fc3 = register_module("fc3", torch::nn::Linear(950, 1050));
        fc4 = register_module("fc4", torch::nn::Linear(1050, 1150));
        fc5 = register_module("fc5", torch::nn::Linear(1150, 1275));
    }

    torch::Tensor forward(torch::Tensor obs, torch::Tensor action_mask)
    {
        obs = torch::relu(fc1->forward(obs));
        obs = torch::relu(fc2->forward(obs));
        obs = torch::relu(fc3->forward(obs));
        obs = torch::relu(fc4->forward(obs));
        torch::Tensor logits = torch::relu(fc5->forward(obs));
        auto masked_logits = torch::where(action_mask > 0, logits, -100000000.0);
        cout << masked_logits << endl;
        // Create categorical distribution of masked logits
        auto logsumexp = torch::logsumexp(masked_logits, -1, true);
        cout << logsumexp << endl;
        masked_logits = masked_logits - logsumexp;
        cout << masked_logits << endl;
        auto probs = torch::softmax(masked_logits, -1);
        cout << probs << endl;
        // Sample the action
        auto action = torch::multinomial(probs, 1);
        cout << action << endl;
        // Get the log probability of the action
        torch::Tensor tensorlist[2] = {torch::unsqueeze(action, -1), masked_logits};
        auto broadcast_result = torch::broadcast_tensors(tensorlist);
        auto value = broadcast_result[0];
        auto log_pmf = broadcast_result[1];
        // cout << value << endl;
        // value = value [..., :1];
        // return log_pmf.gather(-1, value).squeeze(-1);
        return action;
    }

    torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr, fc4 = nullptr, fc5 = nullptr;
};

struct Critic : torch::nn::Module
{
    Critic()
    {
        fc1 = register_module("fc1", torch::nn::Linear(750, 450));
        fc2 = register_module("fc2", torch::nn::Linear(450, 250));
        fc3 = register_module("fc3", torch::nn::Linear(250, 1));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct ActorCritic : torch::nn::Module
{
    ActorCritic()
    {
        Actor pi;
        Critic v;
    }

    torch::Tensor step(torch::Tensor obs, torch::Tensor action_mask)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor ml = pi.forward(obs, action_mask);
        return ml;
    }

    Actor pi = Actor();
    Critic v = Critic();
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
    // torch::manual_seed(42);
    // ActorCritic ac = ActorCritic();
    // torch::load(ac.pi, "C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\actor.pt");
    // torch::Tensor random_obs = torch::rand({750});
    // cout << random_obs << endl;
    // torch::Tensor random_mask = torch::where(torch::rand({1275}) > 0.7, 0, 1);
    // auto ml = ac.step(random_obs, random_mask);
    // cout << ml << endl;
    // return 0;

    // Test torchscript
    torch::jit::script::Module module;
    module = torch::jit::load("C:\\Users\\Jesse\\Projects\\tak_cpp\\data\\traced_ac.pt");
    torch::Tensor random_obs = torch::rand({750});
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(random_obs);
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output << '\n';
    return 0;
}
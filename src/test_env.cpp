#include "env.cpp"
#include <vector>

int main()
{
    // string player_ac;
    // string opponent_ac;
    // cin >> player_ac;
    // cin >> opponent_ac;

    string player_actor = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\traced\\actor.pt";
    string opponent_actor = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\traced\\actor_opponent.pt";

    torch::NoGradGuard no_grad;
    torch::jit::script::Module actor = torch::jit::load(player_actor);
    torch::jit::script::Module actor_opp = torch::jit::load(opponent_actor);

    TakEnvTest env = TakEnvTest();
    vector<char> wins;
    // Play 20 games
    bool opponent_starts = false;
    // 10 games where player starts
    for (int i = 0; i < 10; i++)
    {
        char winner = 'C';
        while (winner == 'C')
        {
            winner = env.step(opponent_starts, actor, actor_opp);
        }
        wins.push_back(winner);
        env.reset();
    }
    // 10 games where opponent starts
    opponent_starts = true;
    for (int i = 0; i < 10; i++)
    {
        char winner = 'C';
        while (winner == 'C')
        {
            winner = env.step(opponent_starts, actor, actor_opp);
        }
        wins.push_back(winner);
        env.reset();
    }
    for (auto c : wins)
    {
        std::cout << c;
        std::cout << ',';
    }
    return 0;
}
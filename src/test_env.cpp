#include "env.cpp"
#include <vector>

int main()
{
    string player_ac;
    string opponent_ac;
    cin >> player_ac;
    cin >> opponent_ac;
    TakEnvTest env = TakEnvTest(player_ac, opponent_ac);
    vector<char> wins;
    // Play 20 games
    bool opponent_starts = false;
    // 10 games where player starts
    for (int i = 0; i < 10; i++)
    {
        char winner = 'C';
        while (winner == 'C')
        {
            winner = env.step(opponent_starts);
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
            winner = env.step(opponent_starts);
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
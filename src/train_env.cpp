#include "env.cpp"
#include <string>

int main()
{
    string ac_filename;
    cin >> ac_filename;
    int games_to_play;
    cin >> games_to_play;
    TakEnv env = TakEnv(ac_filename);
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
#include "env.cpp"
#include <string>
#include <conio.h>

int main()
{
    // string ac_filename;
    // cin >> ac_filename;
    int games_to_play = 1;
    // cin >> games_to_play;

    string actor_filename = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\traced\\actor.pt";
    string critic_filename = "C:\\Users\\Jesse\\Projects\\tak_cpp\\agents\\traced\\critic.pt";

    torch::NoGradGuard no_grad;
    torch::jit::script::Module actor = torch::jit::load(actor_filename);
    torch::jit::script::Module critic = torch::jit::load(critic_filename);
    // player.save("module_1_cpp.pt");
    TakEnv env = TakEnv();
    for (int i = 0; i < games_to_play; i++)
    {
        bool game_ends = false;
        while (!game_ends)
        {
            game_ends = env.step(actor, critic);
        }
        env.reset();
    }
    env.write_experience_to_cout();
}
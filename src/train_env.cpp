#include "env.cpp"
#include <string>
#include <conio.h>

int main()
{
    string ac_filename;
    cin >> ac_filename;
    int games_to_play;
    cin >> games_to_play;

    // string ac_filename = "C:\\Users\\Jesse\\Projects\\tak_cpp\\traced_ac_cpu.pt";
    // int games_to_play = 1;

    torch::NoGradGuard no_grad;
    torch::jit::script::Module player = torch::jit::load(ac_filename);
    player.to(at::kCPU);
    player.eval();
    // player.save("module_1_cpp.pt");
    TakEnv env = TakEnv();
    for (int i = 0; i < games_to_play; i++)
    {
        bool game_ends = false;
        while (!game_ends)
        {
            game_ends = env.step(player);
        }
        env.reset();
    }
    env.write_experience_to_cout();
}
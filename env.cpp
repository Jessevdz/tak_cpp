#include "board.h"

struct GameState
{
    // The NEW state of the board due to the player taking an action
    vector<int> observation;
    // Contains reward: The reward as a result of taking the action.
    // Contains termination info: whether the game is done.
    // If game is done, we need to call reset()
    WinConditions win_conditions;
};

/****************************************
A Tak environment has two players, which
are controlled by policies, and a board.
****************************************/
class TakEnv
{
private:
    Board board = Board();

public:
    void reset();
    GameState step(int);
};

/****************************************
Reset the board state, and return the
initial observation.
****************************************/
void TakEnv::reset()
{
    board = Board();
}

/****************************************
Take an action on the board, and return
an observation, reward, and termination
info.
****************************************/
// GameState TakEnv::step(int action)
// {
//     WinConditions win_conditions = board.take_action(action);
// }

// NEXT: https://pytorch.org/tutorials/advanced/cpp_frontend.html
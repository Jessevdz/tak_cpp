#include "board.h"
#include <stdexcept>

// Move::Move(MoveType type)
// {
//     type = type;
// }

int Board::do_move(const Move &move)
/*
Execute a move on the board.
Return 1 if the game ends and the active player has won.
Return 0 in all other cases.
*/
{
    switch (move.type)
    {
    case PLACE:
        //
        break;
    case MOVE:
        //
        break;
    default:
        throw std::invalid_argument("Board received an invalid move type.");
        break;
    }
};

int main()
{
    Rank r = Rank::ONE;
    int r_value = static_cast<int>(Rank::ONE);
    File f = File::A;
    int f_value = static_cast<int>(File::A);
    Square square = Square();

    PlaceStone move = PlaceStone(File::A, Rank::ONE, StoneType::FLAT);
    Board board = Board();
    board.do_move(move);
}
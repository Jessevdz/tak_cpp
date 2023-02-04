#include <stdexcept>
#include <string>
#include "board.h"

using namespace std;

/****************
UTILITY FUNCTIONS
****************/
string player_as_string(const Player &player)
{
    switch (player)
    {
    case Player::WHITE:
        return "white";
    case Player::BLACK:
        return "black";
    default:
        return "";
    }
};

/**************
BOARD FUNCTIONS
**************/
bool Board::player_has_capstone(const Player &player)
/*
Check if player has a capstone in their reserve.
*/
{
    if (player == Player::WHITE)
    {
        return white_capstone > 0;
    }
    else
    {
        return black_capstone > 0;
    }
};

bool Board::player_has_stones(const Player &player)
/*
Check if player has stones left in their reserve.
*/
{
    if (player == Player::WHITE)
    {
        return white_stone_reserve > 0;
    }
    else
    {
        return black_stone_reserve > 0;
    }
};

Stone Board::take_capstone(const Player &player)
/*
Take a player's capstone from their reserve.
*/
{
    if (player == Player::WHITE)
    {
        white_capstone = 0;
    }
    else
    {
        black_capstone = 0;
    }
    return Stone(player, StoneType::CAPSTONE);
};

Stone Board::take_stone(const Player &player, const char &stone_type)
/*
Take a regular stone from the player's reserve.
*/
{
    if (player == Player::WHITE)
    {
        white_stone_reserve--;
    }
    else
    {
        black_stone_reserve--;
    }
    if (stone_type == 'F')
    {
        return Stone(player, StoneType::FLAT);
    }
    else
    {
        return Stone(player, StoneType::STANDING);
    }
};

Stone Board::take_stone_from_reserve(const char &stone_type)
/*
Take a stone from a player's reserve
*/
{
    const Player &active_player = get_active_player();

    if (stone_type == 'C')
    {
        if (player_has_capstone(active_player))
        {
            return take_capstone(active_player);
        }
        else
        {
            string player_string = player_as_string(active_player);
            throw runtime_error("The " + player_string + " player's capstone is already on the board.");
        }
    }
    else if (stone_type == 'F' | stone_type == 'S')
    {
        if (player_has_stones(active_player))
        {
            return take_stone(active_player, stone_type);
        }
        else
        {
            string player_string = player_as_string(active_player);
            throw runtime_error("The " + player_string + " player has no more stones in their reserve.");
        }
    }
};

int Board::do_move(const string &ptn_move)
/*
Parse a PTN string and execute the move on the board.
Switch active player when move is done.
Return 1 if the game ends and the active player has won.
Return 0 in all other cases.
*/
{
    if (ptn_move.size() == 3)
    {
        // Place stone in an empty square
        char stone_type = ptn_move.at(0);
        char file = ptn_move.at(1);
        char rank = ptn_move.at(2);
        Stone stone = take_stone_from_reserve(stone_type);
    }
    else if (ptn_move.size() == 9)
    {
        // Move a stack of stones
    }
    else
    {
        throw invalid_argument("Board received an invalid PTN move string.");
    }
};

int main()
{
    string place_stone_1 = "CA1";
    string place_stone_2 = "FA2";
    string place_stone_3 = "SA3";
    string move_stack = "1A1+1000F";
    Board board = Board();
    board.do_move(place_stone_1);
    board.do_move(place_stone_2);
    board.do_move(place_stone_3);
}
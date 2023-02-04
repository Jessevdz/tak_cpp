#include <stdexcept>
#include <string>
#include "board.h"

using namespace std;

/****************
UTILITY FUNCTIONS
****************/
string player_as_string(const char &player)
{
    switch (player)
    {
    case 'W':
        return "white";
    case 'B':
        return "black";
    default:
        throw runtime_error("Player has been defined incorrectly.");
    }
};

/**************
BOARD FUNCTIONS
**************/
bool Board::player_has_capstone(const char &player)
/*
Check if player has a capstone in their reserve.
*/
{
    if (player == 'W')
    {
        return white_capstone > 0;
    }
    else if (player == 'B')
    {
        return black_capstone > 0;
    }
    else
    {
        throw runtime_error("Player has been defined incorrectly.");
    }
};

bool Board::player_has_stones(const char &player)
/*
Check if player has stones left in their reserve.
*/
{
    if (player == 'W')
    {
        return white_stone_reserve > 0;
    }
    else if (player == 'B')
    {
        return black_stone_reserve > 0;
    }
    else
    {
        throw runtime_error("Player has been defined incorrectly.");
    }
};

Stone Board::take_capstone(const char &player)
/*
Take a player's capstone from their reserve.
*/
{
    if (player == 'W')
    {
        white_capstone = 0;
    }
    else if (player == 'B')
    {
        black_capstone = 0;
    }
    else
    {
        throw runtime_error("Player has been defined incorrectly.");
    }
    return Stone(player, 'C');
};

Stone Board::take_stone(const char &player, const char &stone_type)
/*
Take a regular stone from the player's reserve.
*/
{
    if (player == 'W')
    {
        white_stone_reserve--;
    }
    else if (player == 'B')
    {
        black_stone_reserve--;
    }
    else
    {
        throw runtime_error("Player has been defined incorrectly.");
    }
    return Stone(player, stone_type);
};

Stone Board::take_stone_from_reserve(const char &stone_type)
/*
Take a stone from a player's reserve
*/
{
    const char &active_player = get_active_player();

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
    else
    {
        throw runtime_error("Attempted to retrieve incorrect stone type from reserve.");
    }
};

void Board::place_stone(const int &file_index, const int &rank_index, const Stone stone)
/*
Place a stone on top of the stack of stones at the square located at [file, rank]
*/
{
    squares[file_index][rank_index].add_stone(stone);
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
        int file_index = file_to_index[file];
        int rank_index = rank_to_index[rank];
        if (!squares[file_index][rank_index].is_empty())
        {
            throw runtime_error("Cannot place a new stone on a square that is not empty.");
        }
        Stone stone = take_stone_from_reserve(stone_type);
        place_stone(file_index, rank_index, stone);
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

void test_placing_stones()
{
    Board board = Board();
    string place_stone_1 = "CA1";
    string place_stone_2 = "FA2";
    string place_stone_3 = "SA3";
    string place_stone_4 = "CA4";
    board.do_move(place_stone_1);
    board.do_move(place_stone_2);
    board.do_move(place_stone_3);
    board.do_move(place_stone_4);
}

void test_moving_stones()
{
    string move_stack = "1A1+1000F";
}

int main()
{
    test_placing_stones();
}
#include <stdexcept>
#include <iostream>
#include <string>
#include <map>
#include "board.h"
#include "util.h"

using namespace std;

/**************
SQUARE FUNCTIONS
**************/

/*********************************
Add a stone to the stack.
*********************************/
void Square::add_stone(Stone stone)
{
    if (!is_empty())
    {
        char &top_stone_type = stones.top().get_type();
        if (top_stone_type == 'S' | top_stone_type == 'C')
        {
            throw runtime_error("Attempting to place a stone on top of a standing- or capstone.");
        }
    }
    stones.push(stone);
}

/*********************************
Remove the top stone and return it.
**********************************/
Stone Square::get_stone()
{
    if (is_empty())
    {
        throw runtime_error("Attempting to grab a stone from an empty square.");
    }
    Stone stone = stones.top();
    stones.pop();
    return stone;
}

/**************
BOARD FUNCTIONS
**************/
Board::Board()
{
    all_ptn_moves = enumerate_all_ptn_moves();
}

/********************************************************
Check if player has a capstone in their reserve.
********************************************************/
bool Board::player_has_capstone(const char &player)
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

/********************************************************
Check if player has stones left in their reserve.
********************************************************/
bool Board::player_has_stones(const char &player)
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

/********************************************************
Take a player's capstone from their reserve.
********************************************************/
Stone Board::take_capstone(const char &player)
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

/********************************************************
Take a regular stone from the player's reserve.
********************************************************/
Stone Board::take_stone(const char &player, const char &stone_type)
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

/********************************************************
Take a stone from a player's reserve.
********************************************************/
Stone Board::take_stone_from_reserve(const char &stone_type)
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

/********************************************************
TODO Remove eventually.
Place a stone on top of the stack of stones
at the square located at [file, rank]
********************************************************/
void Board::place_stone(const int &file_index, const int &rank_index, const Stone stone)
{
    squares[file_index][rank_index].add_stone(stone);
};

/********************************************************
Parse a PTN string and change the board state accordingly.
********************************************************/
void Board::execute_ptn_move(const string &ptn_move)
{
    if (ptn_move.size() == 3)
    // Place stone in an empty square
    {
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
        squares[file_index][rank_index].add_stone(stone);
    }
    else if (ptn_move.size() == 8)
    // Move a stack of stones
    {
        int stones_to_take = ptn_move.at(0) - '0'; // Converts char representation to int
        if (stones_to_take > 5 | stones_to_take < 1)
        {
            throw runtime_error("Players can only move stacks of at least 1 and at most 5 stones on a 5x5 board.");
        }
        char file = ptn_move.at(1);
        char rank = ptn_move.at(2);
        int start_file_index = file_to_index[file];
        int start_rank_index = rank_to_index[rank];
        int cur_file_index = start_file_index;
        int cur_rank_index = start_rank_index;
        char direction = ptn_move.at(3);
        string drop_counts = ptn_move.substr(4, 8);

        stack<Stone> moving_stones;
        while (stones_to_take > 0)
        {
            moving_stones.push(squares[start_file_index][start_rank_index].get_stone());
            stones_to_take--;
        }
        for (auto it = drop_counts.cbegin(); *it != '0'; ++it)
        {
            int stones_to_drop = *it - '0'; // Converts char representation to int
            switch (direction)
            {
            case '+':
                // Rank increases
                cur_rank_index++;
                break;
            case '-':
                // Rank decreases
                cur_rank_index--;
                break;
            case '<':
                // File decreases
                cur_file_index--;
                break;
            case '>':
                // File increases
                cur_file_index++;
                break;
            }
            if ((cur_file_index > 5 | cur_file_index < 0) | (cur_rank_index > 5 | cur_rank_index < 0))
            {
                throw runtime_error("Attempting to place a stone outside of the board.");
            }
            while (stones_to_drop > 0)
            {
                squares[cur_file_index][cur_rank_index].add_stone(moving_stones.top());
                moving_stones.pop();
                stones_to_drop--;
            }
        }
    }
    else
    {
        throw invalid_argument("Board received an invalid PTN move string.");
    }
};

int Board::do_move(const string &ptn_move)
/*
Execute the PTN move
Check end-of-game criteria
Switch the active player
Return 1 if the game ends, 0 if it does not.
*/
{
    return 1;
};

void test_placing_stones()
{
    Board board = Board();
    string place_stone_1 = "CA1";
    string place_stone_2 = "FA2";
    string place_stone_3 = "SA3";
    string place_stone_4 = "CA4";
    board.execute_ptn_move(place_stone_1);
    board.execute_ptn_move(place_stone_2);
    board.execute_ptn_move(place_stone_3);
    board.execute_ptn_move(place_stone_4);
}

void test_moving_stones()
{
    Board board = Board();
    // Set up stack
    board.place_stone(2, 1, Stone('W', 'F'));
    board.place_stone(2, 1, Stone('B', 'F'));
    board.place_stone(2, 1, Stone('W', 'F'));
    board.place_stone(2, 1, Stone('B', 'F'));
    board.place_stone(2, 1, Stone('W', 'C'));
    string move_stack = "4C2+2110";
    board.execute_ptn_move(move_stack);
}

int main()
{
    // test_placing_stones();
    test_moving_stones();
}
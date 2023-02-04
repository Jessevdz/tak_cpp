#include <stdexcept>
#include <iostream>
#include <string>
#include <algorithm>
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

map<int, string> enumerate_all_ptn_moves()
{
    // Mapping between ranks and files, and their integer positions
    map<string, int> rank_to_index = {
        {"1", 1},
        {"2", 2},
        {"3", 3},
        {"4", 4},
        {"5", 5},
    };
    map<string, int> file_to_index = {
        {"A", 1},
        {"B", 2},
        {"C", 3},
        {"D", 4},
        {"E", 5},
    };
    // Mapping between amount of stones moved and valid drop counts.
    vector<string>
        drop_counts_one{"1000"};
    vector<string> drop_counts_two{"2000", "1100"};
    vector<string> drop_counts_three{"3000", "2100", "1200", "1110"};
    vector<string> drop_counts_four{
        "4000",
        "3100",
        "2200",
        "2110",
        "1300",
        "1210",
        "1120",
        "1111",
    };
    vector<string> drop_counts_five{
        "5000",
        "4100",
        "3200",
        "3110",
        "2300",
        "2210",
        "2120",
        "2111",
        "1400",
        "1310",
        "1220",
        "1211",
        "1130",
        "1121",
        "1112",
    };
    map<string, vector<string>>
        valid_drop_counts{
            {"1", drop_counts_one},
            {"2", drop_counts_two},
            {"3", drop_counts_three},
            {"4", drop_counts_four},
            {"5", drop_counts_five},
        };
    vector<string> file{"A", "B", "C", "D", "E"};
    vector<string> rank{"1", "2", "3", "4", "5"};
    vector<string> stone_type{"C", "F", "S"};
    vector<string> nr_of_stones{"1", "2", "3", "4", "5"};
    vector<string> directions{"+", "-", "<", ">"};
    vector<string> all_ptn_moves;
    // Enumerate all placement moves
    for (const string r : rank)
    {
        for (const string f : file)
        {
            for (const string s : stone_type)
            {
                string placement_move = s + f + r;
                all_ptn_moves.push_back(placement_move);
            }
        }
    }
    // Enumerate all stack movements
    for (string n : nr_of_stones)
    {
        vector<string> drop_counts = valid_drop_counts[n];
        for (const string r : rank)
        {
            for (const string f : file)
            {
                for (const string dr : directions)
                {
                    for (const string dc : drop_counts)
                    {
                        // Depending on where we are in the board, some moves are invalid
                        string dc_copy = dc;
                        // Remove all '0' from drop count string
                        dc_copy.erase(remove(dc_copy.begin(), dc_copy.end(), '0'), dc_copy.end());
                        int n_squares = dc_copy.size(); // Amount of squares we want to move in a direction
                        if (dr == ">")
                        {
                            int file_index = file_to_index[f];
                            if ((file_index + n_squares) >= 5)
                                continue;
                        }
                        if (dr == "<")
                        {
                            int file_index = file_to_index[f];
                            if ((file_index - n_squares) < 1)
                                continue;
                        }
                        if (dr == "+")
                        {
                            int rank_index = rank_to_index[r];
                            if ((rank_index + n_squares) >= 5)
                                continue;
                        }
                        if (dr == "-")
                        {
                            int rank_index = rank_to_index[r];
                            if ((rank_index - n_squares) < 1)
                                continue;
                        }
                        string placement_move = n + f + r + dr + dc;
                        all_ptn_moves.push_back(placement_move);
                    }
                }
            }
        }
    }
    // Create int->ptn move mapping
    map<int, string> all_ptn_moves_map;
    int i = 0;
    for (string m : all_ptn_moves)
    {
        all_ptn_moves_map[i] = m;
        i++;
    }
    return all_ptn_moves_map;
}

/**************
SQUARE FUNCTIONS
**************/
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
TODO Remove eventually.
Place a stone on top of the stack of stones at the square located at [file, rank]
*/
{
    squares[file_index][rank_index].add_stone(stone);
};

void Board::execute_ptn_move(const string &ptn_move)
/*
Parse a PTN string and change the board state accordingly.
*/
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
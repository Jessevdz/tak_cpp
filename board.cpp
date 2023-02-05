#include <stdexcept>
#include <iostream>
#include <string>
#include <map>
#include "util.h"
#include "board.h"

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
        char &top_stone_type = stones.top().type;
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

/************************************************************************
Check if the top piece of the stack is elligible for a road for a player.
************************************************************************/
bool Square::is_road_square(const char &active_player)
{
    if (is_empty())
    {
        return false;
    }
    Stone &stone = stones.top();
    if ((stone.color == active_player) && ((stone.type == 'F') | ((stone.type == 'C'))))
    {
        return true;
    }
    return false;
}

/********************************************************
Return true if the top piece belongs to the active player
********************************************************/
bool Square::is_controlled_by(const char &active_player)
{
    if (is_empty())
        return false;
    return (stones.top().color == active_player);
}

/********************************************************
Return true if the top piece is blocking wrt. moving a stack
********************************************************/
bool Square::is_blocking()
{
    if (is_empty())
        return false;
    if ((stones.top().type == 'S') | (stones.top().type == 'C'))
    {
        return true;
    }
    return false;
}

/********************************************************
Return the type char of the top stone in this square.
********************************************************/
char Square::get_top_stone_type()
{
    if (is_empty())
        return 'E';
    return stones.top().type;
}

/**************
BOARD FUNCTIONS
**************/
Board::Board()
{
    int_to_ptn_move = enumerate_all_ptn_moves();

    for (auto const &m : int_to_ptn_move)
    {
        ptn_move_to_int[m.second] = m.first;
    }
}

/********************************************************
Check if active player has a capstone in their reserve.
********************************************************/
bool Board::player_has_capstone()
{
    if (active_player == 'W')
    {
        return white_capstone > 0;
    }
    else if (active_player == 'B')
    {
        return black_capstone > 0;
    }
    else
    {
        throw runtime_error("Active player has been defined incorrectly.");
    }
};

/********************************************************
Check if active player has stones left in their reserve.
********************************************************/
bool Board::player_has_stones()
{
    if (active_player == 'W')
    {
        return white_stone_reserve > 0;
    }
    else if (active_player == 'B')
    {
        return black_stone_reserve > 0;
    }
    else
    {
        throw runtime_error("Active player has been defined incorrectly.");
    }
};

/********************************************************
Check if the active player has a road on the board.
********************************************************/
bool Board::player_has_road()
{
    // Create a road array. This is an array with the same size as the board.
    // It contains 1 if the square is eligible for a road, i.e.,
    // the square contains a flat piece of the capstone on the top of the stack.
    // It contains a 0 otherwise.
    int road_array[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };

    // We can end the search early by checking whether there are road pieces
    // both at the top and bottom of the board, or both at the left and right.
    // Otherwise, a road is not possible.
    bool top = false, bottom = false, left = false, right = false;
    int bottom_edge = 0;
    int top_edge = 4;
    for (int i = 0; i < 5; i++)
    {
        if (squares[bottom_edge][i].is_road_square(active_player)) // Checks left edge bottom-top
        {
            road_array[bottom_edge][i] = 1;
            left = true;
        }
        if (squares[top_edge][i].is_road_square(active_player)) // Checks right edge bottom-top
        {
            road_array[top_edge][i] = 1;
            right = true;
        }
        if (squares[i][top_edge].is_road_square(active_player)) // Checks top edge left-right
        {
            road_array[i][top_edge] = 1;
            top = true;
        }
        if (squares[i][bottom_edge].is_road_square(active_player)) // Checks bottom edge left-right
        {
            road_array[i][bottom_edge] = 1;
            bottom = true;
        }
    }
    if (!((top && bottom) | (left && right)))
    {
        // A road is not possible
        return false;
    }

    // A road might be possible, so continue building the road array.
    for (int file = 1; file < 4; file++)
    {
        for (int rank = 1; rank < 4; rank++)
        {
            if (squares[file][rank].is_road_square(active_player))
            {
                road_array[file][rank] = 1;
            }
        }
    }

    if (find_road(road_array))
    {
        return true;
    }
    else
    {
        return false;
    }
}

/********************************************************
Take the active player's capstone from their reserve.
********************************************************/
Stone Board::take_capstone()
{
    if (active_player == 'W')
    {
        white_capstone = 0;
    }
    else if (active_player == 'B')
    {
        black_capstone = 0;
    }
    else
    {
        throw runtime_error("Active player has been defined incorrectly.");
    }
    return Stone(active_player, 'C');
};

/********************************************************
Take a regular stone from the active player's reserve.
********************************************************/
Stone Board::take_stone(const char &stone_type)
{
    if (active_player == 'W')
    {
        white_stone_reserve--;
    }
    else if (active_player == 'B')
    {
        black_stone_reserve--;
    }
    else
    {
        throw runtime_error("Active player has been defined incorrectly.");
    }
    return Stone(active_player, stone_type);
};

/********************************************************
Take a stone from the active player's reserve.
********************************************************/
Stone Board::take_stone_from_reserve(const char &stone_type)
{
    if (stone_type == 'C')
    {
        if (player_has_capstone())
        {
            return take_capstone();
        }
        else
        {
            string player_string = player_as_string(active_player);
            throw runtime_error("The " + player_string + " player's capstone is already on the board.");
        }
    }
    else if (stone_type == 'F' | stone_type == 'S')
    {
        if (player_has_stones())
        {
            return take_stone(stone_type);
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

/*************************************************************
Return a mask indicating valid PTN moves for the active player.
*************************************************************/
vector<int> Board::valid_moves()
{
    // Gather necessary state from the board

    // Array indicating which squares the active player controls
    // It contains 1 if the active player controls the square
    // It contains 0 if the active player does not control the square
    int control_array[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };

    // Array indicating which squares contain which types of stones on top of the stack
    // Contains "E" if the square is empty.
    char stone_type_array[5][5] = {
        {'E', 'E', 'E', 'E', 'E'},
        {'E', 'E', 'E', 'E', 'E'},
        {'E', 'E', 'E', 'E', 'E'},
        {'E', 'E', 'E', 'E', 'E'},
        {'E', 'E', 'E', 'E', 'E'},
    };

    // File-to-rank map of squares that contain blocking pieces.
    map<int, int> blocking_files;
    map<int, int> blocking_ranks; // Rank-to-file

    // Populate the arrays
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            if (squares[file][rank].is_controlled_by(active_player))
            {
                control_array[file][rank] = 1;
            }
            stone_type_array[file][rank] = squares[file][rank].get_top_stone_type();
            if (squares[file][rank].is_blocking())
            {
                blocking_files[file] = rank;
                blocking_ranks[rank] = file;
            }
        }
    }

    // Check which stones the player can still play
    vector<string> valid_stone_types;
    bool player_has_stones;
    bool player_has_capstone;
    if (active_player == 'W')
    {
        if (white_capstone > 0)
            player_has_capstone = true;
        if (white_stone_reserve > 0)
            player_has_stones = true;
    }
    else
    {
        if (black_capstone > 0)
            player_has_capstone = true;
        if (black_stone_reserve > 0)
            player_has_stones = true;
    }
    if (player_has_capstone && player_has_stones)
    {
        valid_stone_types = {"C", "S", "F"};
    }
    else if (!player_has_capstone && player_has_stones)
    {
        valid_stone_types = {"S", "F"};
    }
    else if (player_has_capstone && !player_has_stones)
    {
        valid_stone_types = {"C"};
    }
    else
    {
        throw runtime_error("Player has no more stones. The game should have ended.");
    }

    // Enumerate all valid PTN moves for active player
    /*
    TODO: Edge case with capstones flattening standing stones.
    */
    vector<string> valid_ptn_moves;
    bool check_blocking_file = false;
    bool check_blocking_rank = false;
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            /*
            If the square is empty, we can place stones there.
            */
            if (stone_type_array[file][rank] == 'E')
            {
                for (const string t : valid_stone_types)
                {
                    string m = t + to_string(file) + to_string(rank);
                    valid_ptn_moves.push_back(m);
                }
            }
            /*
            If we control the square, we can move the stack it contains.
            */
            if (control_array[file][rank] == 1)
            {
                // Amount of stones we can move
                int stack_size = squares[file][rank].get_size();
                if (stack_size > 5)
                    stack_size = 5;
                // Are there any blocking stones in this file and rank?
                if (blocking_files.find(file) != blocking_files.end())
                {
                    check_blocking_file = true;
                }
                if (blocking_ranks.find(rank) != blocking_ranks.end())
                {
                    check_blocking_rank = true;
                }
                int movement_squares;
                for (string dr : _directions)
                {
                    // Figure out how far we can move in each direction
                    movement_squares = 0;
                    if (dr == "+")
                    {
                        if (check_blocking_file)
                        {
                            int blocking_rank = blocking_files[file];
                            movement_squares = 4 - blocking_rank - 1;
                        }
                        else
                        {
                            movement_squares = 4 - rank;
                        }
                    }
                    if (dr == "-")
                    {
                        if (check_blocking_file)
                        {
                            int blocking_rank = blocking_files[file];
                            movement_squares = rank - blocking_rank - 1;
                        }
                        else
                        {
                            movement_squares = rank;
                        }
                    }
                    if (dr == ">")
                    {
                        if (check_blocking_rank)
                        {
                            int blocking_file = blocking_ranks[rank];
                            movement_squares = blocking_file - 1;
                        }
                        else
                        {
                            movement_squares = 4 - file;
                        }
                    }
                    if (dr == "<")
                    {
                        if (check_blocking_rank)
                        {
                            int blocking_file = blocking_ranks[rank];
                            movement_squares = file - blocking_file - 1;
                        }
                        else
                        {
                            movement_squares = file;
                        }
                    }

                    if (movement_squares <= 0)
                    {
                        continue;
                    }
                    else
                    {
                        // Enumerate the movement strings
                        // stack_size == how many stones we can move from this position
                        // movement_squares == how many squares we can move them in
                        for (int nr_stones = 0; nr_stones <= stack_size; nr_stones++)
                        {
                            for (int nr_sq = 0; nr_sq <= movement_squares; nr_sq++)
                            {
                                vector<string> drop_counts = _valid_drop_counts_move_squares[nr_stones][nr_sq];
                                for (string dc : drop_counts)
                                {
                                    string m = to_string(nr_stones) + _index_to_file[file] + to_string(rank + 1) + dr + dc;
                                    valid_ptn_moves.push_back(m);
                                }
                            }
                        }
                    }
                }
                // safety
                check_blocking_file = false;
                check_blocking_rank = false;
            }
        }
    }
    vector<int> ret;
    return ret;
}
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

/**************
TEST FUNCTIONS
**************/
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

void test_find_vertical_road()
{
    // There is a road
    Board board = Board();
    board.place_stone(0, 0, Stone('W', 'F'));
    board.place_stone(1, 0, Stone('W', 'F'));
    board.place_stone(1, 1, Stone('W', 'F'));
    board.place_stone(2, 1, Stone('W', 'F'));
    board.place_stone(3, 1, Stone('W', 'C'));
    board.place_stone(3, 2, Stone('W', 'F'));
    board.place_stone(3, 3, Stone('W', 'F'));
    board.place_stone(3, 4, Stone('W', 'F'));
    if (board.player_has_road())
    {
        cout << "Player has a road." << endl;
    }
    else
    {
        cout << "Player does not have a road." << endl;
    }
}

void test_find_horizontal_road()
{
    // There is a road
    Board board = Board();
    board.place_stone(0, 4, Stone('W', 'F'));
    board.place_stone(1, 4, Stone('W', 'F'));
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'C'));
    board.place_stone(2, 2, Stone('W', 'F'));
    board.place_stone(3, 2, Stone('W', 'F'));
    board.place_stone(4, 2, Stone('W', 'F'));
    if (board.player_has_road())
    {
        cout << "Player has a road." << endl;
    }
    else
    {
        cout << "Player does not have a road." << endl;
    }
}

void test_find_road_blocked()
{
    // There is a road
    Board board = Board();
    board.place_stone(0, 4, Stone('W', 'F'));
    board.place_stone(1, 4, Stone('W', 'F'));
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'C'));
    board.place_stone(2, 2, Stone('W', 'F'));
    board.place_stone(3, 2, Stone('W', 'S'));
    board.place_stone(4, 2, Stone('W', 'F'));
    if (board.player_has_road())
    {
        cout << "Player has a road." << endl;
    }
    else
    {
        cout << "Player does not have a road." << endl;
    }
}

void test_find_moves()
{
    // There is a road
    Board board = Board();
    board.place_stone(0, 4, Stone('W', 'F'));
    board.place_stone(1, 4, Stone('W', 'S'));
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'C'));
    board.place_stone(2, 2, Stone('W', 'F'));
    board.place_stone(3, 2, Stone('W', 'S'));
    board.place_stone(4, 2, Stone('W', 'F'));
    board.valid_moves();
}

int main()
{
    // test_placing_stones();
    // test_moving_stones();
    // test_find_vertical_road();
    // test_find_horizontal_road();
    // test_find_road_blocked();
    test_find_moves();
}
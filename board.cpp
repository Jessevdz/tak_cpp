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
        if (top_stone_type == 'C')
        {
            throw runtime_error("Attempting to place a stone on top of a capstone.");
        }
        if (top_stone_type == 'S')
        {
            if (stone.type == 'C')
            {
                // Flatten standing stone
                stones.top().flatten();
            }
            else
            {
                throw runtime_error("Attempting to place a stone on top of a standing stone.");
            }
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

/*********************************
Return a reference to the top stone
**********************************/
const Stone &Square::peek_top_stone()
{
    const Stone &stone = stones.top();
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

/********************************************************
Transform the square to a vector representation.
Only the top-10 stones are considered in the representation.
********************************************************/
vector<int> Square::get_square_state(const char active_player)
{
    stack<Stone> s = stones;
    vector<int> square_state;
    for (int i = 0; i < 10; i++)
    {
        if (s.empty())
        {
            vector<int> empty_space = {0, 0, 0};
            square_state.insert(square_state.end(), empty_space.begin(), empty_space.end());
        }
        else
        {
            Stone stone = s.top();
            vector<int> stone_vec;
            if (stone.color == active_player)
            {
                stone_vec = player_stone_type_to_vec[stone.type];
            }
            else
            {
                stone_vec = opponent_stone_type_to_vec[stone.type];
            }
            square_state.insert(square_state.end(), stone_vec.begin(), stone_vec.end());
            s.pop();
        }
    }
    return square_state;
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
Check if it is the active player's first move
********************************************************/
bool Board::player_first_move()
{
    if (active_player == 'W')
    {
        return white_capstone + white_stone_reserve == 22;
    }
    else if (active_player == 'B')
    {
        return black_capstone + black_stone_reserve == 22;
    }
    else
    {
        throw runtime_error("Active player has been defined incorrectly.");
    }
};

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
bool Board::player_has_road(const char &player)
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
        if (squares[bottom_edge][i].is_road_square(player)) // Checks left edge bottom-top
        {
            road_array[bottom_edge][i] = 1;
            left = true;
        }
        if (squares[top_edge][i].is_road_square(player)) // Checks right edge bottom-top
        {
            road_array[top_edge][i] = 1;
            right = true;
        }
        if (squares[i][top_edge].is_road_square(player)) // Checks top edge left-right
        {
            road_array[i][top_edge] = 1;
            top = true;
        }
        if (squares[i][bottom_edge].is_road_square(player)) // Checks bottom edge left-right
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
            if (squares[file][rank].is_road_square(player))
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
Check if there are no empty squares on the board.
********************************************************/
bool Board::board_is_full()
{
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            if (squares[file][rank].is_empty())
            {
                return false;
            }
        }
    }
    return true;
}

/********************************************************
Return the player chart hat has the most controlling
flat stones on the board. Return "T" if it is a tie.
********************************************************/
char Board::check_flat_win()
{
    int black = 0, white = 0;
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            if (!squares[file][rank].is_empty())
            {
                const Stone &top_stone = squares[file][rank].peek_top_stone();
                if (top_stone.type == 'F')
                {
                    if (top_stone.color == 'W')
                    {
                        white++;
                    }
                    else
                    {
                        black++;
                    }
                }
            }
        }
    }
    if (black == white)
        return 'T';
    else if (black > white)
        return 'B';
    return 'W';
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
vector<string> Board::valid_moves()
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
    vector<string> valid_ptn_moves;
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
                    string m = t + _index_to_file[file] + to_string(rank + 1);
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
                int movement_squares;
                // Indicate whether the current stone is a capstone, and we can use it to flatten a standing stone in our path
                bool top_is_capstone = (stone_type_array[file][rank] == 'C');
                bool capstone_can_flatten = false;
                for (string dr : _directions)
                {
                    // Figure out how far we can move in each direction
                    movement_squares = 0;
                    if (dr == "+")
                    {
                        int cur_rank = rank;
                        while (cur_rank < 4)
                        {
                            cur_rank++;
                            if (stone_type_array[file][cur_rank] == 'C')
                            {
                                break;
                            }
                            if (stone_type_array[file][cur_rank] == 'S')
                            {
                                if (top_is_capstone)
                                {
                                    movement_squares += 1;
                                    capstone_can_flatten = true;
                                }
                                break;
                            }
                            movement_squares += 1;
                        }
                    }
                    if (dr == "-")
                    {
                        int cur_rank = rank;
                        while (cur_rank > 0)
                        {
                            cur_rank--;
                            if (stone_type_array[file][cur_rank] == 'C')
                            {
                                break;
                            }
                            if (stone_type_array[file][cur_rank] == 'S')
                            {
                                if (top_is_capstone)
                                {
                                    movement_squares += 1;
                                    capstone_can_flatten = true;
                                }
                                break;
                            }
                            movement_squares += 1;
                        }
                    }
                    if (dr == ">")
                    {
                        int cur_file = file;
                        while (cur_file < 4)
                        {
                            cur_file++;
                            if (stone_type_array[cur_file][rank] == 'C')
                            {
                                break;
                            }
                            if (stone_type_array[cur_file][rank] == 'S')
                            {
                                if (top_is_capstone)
                                {
                                    movement_squares += 1;
                                    capstone_can_flatten = true;
                                }
                                break;
                            }
                            movement_squares += 1;
                        }
                    }
                    if (dr == "<")
                    {
                        int cur_file = file;
                        while (cur_file > 0)
                        {
                            cur_file--;
                            if (stone_type_array[cur_file][rank] == 'C')
                            {
                                break;
                            }
                            if (stone_type_array[cur_file][rank] == 'S')
                            {
                                if (top_is_capstone)
                                {
                                    movement_squares += 1;
                                    capstone_can_flatten = true;
                                }
                                break;
                            }
                            movement_squares += 1;
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
                        for (int nr_stones = 1; nr_stones <= stack_size; nr_stones++)
                        {
                            for (int nr_sq = 1; nr_sq <= movement_squares; nr_sq++)
                            {
                                vector<string> drop_counts;
                                if (capstone_can_flatten)
                                {
                                    // Some invalid combinations exist here, trying to drop a stack with a capstone on top of a standing stone
                                    if (nr_sq == 1 & nr_stones != 1)
                                        break;
                                    drop_counts = _valid_drop_counts_move_squares_wcapstone[nr_stones][nr_sq];
                                }
                                else
                                {
                                    drop_counts = _valid_drop_counts_move_squares[nr_stones][nr_sq];
                                }
                                for (string dc : drop_counts)
                                {
                                    string m = to_string(nr_stones) + _index_to_file[file] + to_string(rank + 1) + dr + dc;
                                    valid_ptn_moves.push_back(m);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return valid_ptn_moves;
}

/********************************************************
USED FOR TESTING
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
        Stone stone{'0', '0'};
        // If this is the first move of the game, the active player places a stone from the other player's reserve.
        if (white_first_move)
        {
            switch_active_player();
            stone = take_stone_from_reserve(stone_type);
            switch_active_player();
            white_first_move = false;
        }
        else if (black_first_move)
        {
            switch_active_player();
            stone = take_stone_from_reserve(stone_type);
            switch_active_player();
            black_first_move = false;
        }
        else
        {
            // Default placement
            stone = take_stone_from_reserve(stone_type);
        }
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

        auto it = drop_counts.cbegin(); // iterate over drop count string
        for (int i = 0; i <= 4; ++i)    // There are max 4 drop counts
        {
            if (*it == '0') // End of drop count string
                break;
            if (*it == 0) // End of drop count string iterator
                break;
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
            if ((cur_file_index > 4 | cur_file_index < 0) | (cur_rank_index > 4 | cur_rank_index < 0))
            {
                throw runtime_error("Attempting to place a stone outside of the board.");
            }
            while (stones_to_drop > 0)
            {
                squares[cur_file_index][cur_rank_index].add_stone(moving_stones.top());
                moving_stones.pop();
                stones_to_drop--;
            }
            ++it;
        }
    }
    else
    {
        throw invalid_argument("Board received an invalid PTN move string.");
    }
};

/************************************************************************
Check if the game has ended, and return information on the winning player.
************************************************************************/
WinConditions Board::check_win_conditions()
{
    WinConditions win_conditions;
    // If at least one player is out of stones,
    // or the board has no empty squares, check for a flat win.
    bool white_reserve_empty = (white_stone_reserve + white_capstone == 0);
    bool black_reserve_empty = (black_stone_reserve + black_capstone == 0);
    bool board_full = board_is_full();
    if (white_reserve_empty | black_reserve_empty | board_full)
    {
        char winner = check_flat_win();
        int reward;
        if (winner == 'T')
        {
            reward = 0;
        }
        else
        {
            reward = 1;
        }
        win_conditions = {true, winner, reward, "Flat win"};
        return win_conditions;
    }
    // Both players could have roads
    else if (player_has_road('W'))
    {
        win_conditions = {true, 'W', 1, "Road"};
    }
    else if (player_has_road('B'))
    {
        win_conditions = {true, 'B', 1, "Road"};
    }
    else
    {
        win_conditions = {false, 'T', 0, "Not done."};
    }
    return win_conditions;
}

/********************************************************
Execute the PTN move
Check end-of-game criteria
Switch the active player
Return win conditions
********************************************************/
WinConditions Board::do_move(const string &ptn_move)
{
    execute_ptn_move(ptn_move);
    WinConditions win_conditions = check_win_conditions();
    switch_active_player();
    return win_conditions;
};

/**************************************************************
Execute a move on the board as indicated by its' action integer
**************************************************************/
WinConditions Board::take_action(const int action)
{
    const string ptn_move = int_to_ptn_move[action];
    return do_move(ptn_move);
}

/****************************************************************
Represent the board state as an int vector for the active player.
****************************************************************/
vector<int> Board::get_board_state()
{
    vector<int> board_state;
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            vector<int> square_state = squares[file][rank].get_square_state(active_player);
            board_state.insert(board_state.end(), square_state.begin(), square_state.end()); // Concatenate
        }
    }
    return board_state;
}

/****************************************************************
Represent the board state as an int vector for the active player.
****************************************************************/
void Board::reset()
{
    // GAME STATE VARIABLES
    game_has_ended = false;
    active_player = 'W';
    // Stone counts for players
    white_stone_reserve = 21;
    black_stone_reserve = 21;
    black_capstone = 1;
    white_capstone = 1;
    white_first_move = true;
    black_first_move = true;
}
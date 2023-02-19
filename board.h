#ifndef BOARD_H
#define BOARD_H

#include <stdexcept>
#include <vector>
#include <stack>
#include <map>
#include <string>

using namespace std;

class Stone
{
public:
    Stone(char c, char t) : type{t}, color{c} {}
    char type;
    char color;
    void flatten()
    // Flatten a standing stone into a flat stone.
    {
        if (type == 'S')
        {
            type = 'F';
        }
        else
        {
            throw runtime_error("Cannot flatten anything other than a standing stone.");
        }
    };
};

class Square
{
private:
    stack<Stone> stones;

public:
    bool is_empty() { return stones.empty(); };
    bool is_road_square(const char &);
    bool is_controlled_by(const char &);
    bool is_blocking();
    void add_stone(Stone);
    Stone get_stone();
    const Stone &peek_top_stone();
    char get_top_stone_type();
    int get_size() { return stones.size(); };
    vector<int> get_square_state(const char);
};

struct WinConditions
{
    bool game_ends;
    char winner;
    int reward;
    string win_type;
};

class Board
{
private:
    // GAME STATE VARIABLES
    bool game_has_ended = false;
    char active_player = 'W';
    // Stone counts for players
    int white_stone_reserve = 21;
    int black_stone_reserve = 21;
    int black_capstone = 1;
    int white_capstone = 1;
    bool white_first_move = true;
    bool black_first_move = true;
    // 2D vector containing all squares composing the board.
    // Index file-first.
    vector<vector<Square>> squares{
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
    };
    map<int, string> int_to_ptn_move;
    map<string, int> ptn_move_to_int;
    map<char, int> rank_to_index = {
        {'1', 0},
        {'2', 1},
        {'3', 2},
        {'4', 3},
        {'5', 4},
    };
    map<char, int> file_to_index = {
        {'A', 0},
        {'B', 1},
        {'C', 2},
        {'D', 3},
        {'E', 4},
    };
    bool player_first_move();
    bool player_has_capstone();
    bool player_has_stones();
    bool board_is_full();
    char check_flat_win();
    Stone take_stone_from_reserve(const char &);
    Stone take_capstone();
    Stone take_stone(const char &);
    void switch_active_player()
    {
        if (active_player == 'W')
        {
            active_player = 'B';
        }
        else
        {
            active_player = 'W';
        }
    };
    WinConditions check_win_conditions();

public:
    Board();
    void place_stone(const int &, const int &, const Stone);
    void execute_ptn_move(const string &); // move to private eventually.
    bool player_has_road(const char &);    // move to private eventually.
    vector<string> valid_moves();          // move to private eventually.
    vector<int> get_board_state();
    vector<int> get_valid_moves_mask();
    const char get_active_player() { return active_player; };
    WinConditions do_move(const string &);
    WinConditions take_action(const int);
    void reset_board();
};

#endif // BOARD_H

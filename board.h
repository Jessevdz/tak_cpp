#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <stack>
#include <map>
#include <string>

using namespace std;

/***********************************
DEFINITIONS OF CHARACTER IDENTIFIERS

PLAYER
    W = white player
    B = black player
FILE
    left-to-right columns of the board,
    indicated by A, B, C, D and E
RANK
    bottom-to-top rows of the board,
    indicated by 1, 2, 3, 4 and 5
STONE TYPE
    F = flat stone
    S = standing stone
    C = capstone
STONE COLOR
    Same as the player definition
STACK MOVEMENT
    + = move to higher rows (up the board)
    - = move to lower rows (down the board)
    > = move to higher files (to the right)
    < = move to lower files (to the left)
***********************************/

class Stone
{
private:
    char type;
    char color;

public:
    Stone(char c, char t) : type{t}, color{c} {}
    char &get_type() { return type; };
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
    stack<Stone> stones; // Stones contained on this square

public:
    bool is_empty() { return stones.empty(); };
    void add_stone(Stone stone) { stones.push(stone); };
    Stone get_stone()
    {
        Stone stone = stones.top();
        stones.pop();
        return stone;
    };
};

class Board
{
private:
    bool game_has_ended = false;
    char active_player = 'W';
    // Stone counts for players
    int white_stone_reserve = 21;
    int black_stone_reserve = 21;
    int black_capstone = 1;
    int white_capstone = 1;
    map<int, string> all_ptn_moves; // Mapping from int values to all possible PTN strings.
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
    // 2D vector containing all squares composing the board.
    // Index file-first.
    vector<vector<Square>> squares{
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
    };
    const char &get_active_player() { return active_player; };
    bool player_has_capstone(const char &);
    bool player_has_stones(const char &);
    Stone take_stone_from_reserve(const char &);
    Stone take_capstone(const char &);
    Stone take_stone(const char &, const char &);

public:
    void place_stone(const int &, const int &, const Stone); // move to private eventually.
    void execute_ptn_move(const string &);                   // move to private eventually.
    vector<int> get_legal_moves_for_player(char);
    int do_move(const string &);
};

#endif // BOARD_H

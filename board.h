#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <stack>
#include <map>
#include <string>

using namespace std;

enum Player
{
    WHITE,
    BLACK
};
enum class File : char
{
    A = 'A',
    B = 'B',
    C = 'C',
    D = 'D',
    E = 'E'
};
enum class Rank : char
{
    ONE = '1',
    TWO = '2',
    THREE = '3',
    FOUR = '4',
    FIVE = '5'
};
enum class StoneType : char
{
    FLAT = 'F',
    STANDING = 'S',
    CAPSTONE = 'C'
};

class Stone
{
private:
    StoneType type;
    Player color;

public:
    Stone(Player color, StoneType type)
    {
        color = color;
        type = type;
    }
    StoneType get_type();
    void flatten(); // When standing stones are flattened into flat stones.
};

class Square
{
private:
    // Stones contained on this square
    stack<Stone> stones;

public:
    // Square(Rank, File);
    bool is_empty();
    std::vector<Stone> remove_stones(int amount);
    void add_stones(std::vector<Stone> stones);
};

class Board
{
private:
    bool game_has_ended = false;
    Player active_player = Player::WHITE;
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
    const Player &get_active_player() { return active_player; };
    Stone take_stone_from_reserve(const char &);
    Stone take_capstone(const Player &);
    Stone take_stone(const Player &, const char &);
    bool player_has_capstone(const Player &);
    bool player_has_stones(const Player &);

public:
    vector<int> get_legal_moves_for_player(Player player);
    int do_move(const string &);
};

#endif // BOARD_H

#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <stack>

enum Player
{
    WHITE,
    BLACK
};
enum File
{
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4
};
enum Rank
{
    ONE = 0,
    TWO = 1,
    THREE = 2,
    FOUR = 3,
    FIVE = 4
};
enum StoneType
{
    FLAT,
    STANDING,
    CAPSTONE
};
enum MoveType
{
    // Place a stone on an empty square
    PLACE,
    // Move an existing stack of stones
    MOVE
};

class Move
{
public:
    Move(MoveType type)
    {
        type = type;
    }
    MoveType type;
};

class PlaceStone : public Move
{
public:
    PlaceStone(File file, Rank rank, StoneType stone, MoveType type = MoveType::PLACE) : Move(type)
    {
        file = file;
        rank = rank;
        stone = stone;
    }
    File file;
    Rank rank;
    StoneType Stone;
};

class MoveStack : public Move
{
public:
    MoveStack(MoveType type) : Move(type)
    {
        // Perhaps additional params
    }
};

class Stone
{
private:
    StoneType type;
    Player color;

public:
    StoneType get_type();
    void flatten(); // When standing stones are flattened into flat stones.
};

class Square
{
private:
    // Stones contained on this square
    std::stack<Stone> stones{

    };

public:
    // Square(Rank, File);
    bool is_empty();
    std::vector<Stone> remove_stones(int amount);
    void add_stones(std::vector<Stone> stones);
};

class Board
{
private:
    // std::vector<std::vector<Square>> squares;
public:
    // 2D vector containing all squares composing the board.
    // Index file-first.
    std::vector<std::vector<Square>> squares{
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
        {Square(), Square(), Square(), Square(), Square()},
    };
    std::vector<Move> get_legal_moves_for_player(Player player);
    int do_move(const Move &);
};

#endif // BOARD_H

#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include "board.h"
std::random_device rd;  // Only used once to initialise (seed) engine
std::mt19937 rng(rd()); // Random-number engine used (Mersenne-Twister in this case)

using namespace std;

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
    if (board.player_has_road('W'))
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
    if (board.player_has_road('W'))
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
    if (board.player_has_road('W'))
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
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 3, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'F'));
    board.place_stone(1, 2, Stone('W', 'C'));
    board.place_stone(2, 2, Stone('W', 'S'));
    board.place_stone(3, 2, Stone('W', 'S'));
    board.place_stone(4, 2, Stone('W', 'F'));
    board.valid_moves();
}

void test_play_random_game()
{
    Board board = Board();
    WinConditions win_conditions = {false, 'T'};
    while (!win_conditions.game_ends)
    {
        vector<string> valid_moves = board.valid_moves();
        uniform_int_distribution<int> uni(0, valid_moves.size() - 1); // Guaranteed unbiased
        int random_index = uni(rng);
        string ptn_move;
        ptn_move = valid_moves[random_index];
        win_conditions = board.do_move(ptn_move);
    }
    std::cout << "Game ended via " << win_conditions.win_type << " in favor of " << win_conditions.winner << endl;
}

int main()
{
    // test_placing_stones();
    // test_moving_stones();
    // test_find_vertical_road();
    // test_find_horizontal_road();
    // test_find_road_blocked();
    // test_find_moves();

    for (int i = 0; i < 100; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        test_play_random_game();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << duration.count() << endl;
    }
}
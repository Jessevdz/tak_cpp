#include <string>
#include <stdexcept>
#include <map>
#include <queue>
#include <vector>
#include <algorithm>
#include "util.h"

using namespace std;

/*************************************
Player char identifier to full string
*************************************/
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

/*************************************
Enumerate all legal PTN moves
*************************************/
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
                        // Depending on where we are on the board, some moves are invalid
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

/*************************************
Find a road in a road array.
*************************************/
/*
A road array is an array with the same size as the board.
It contains 1 if the square is eligible for a road, i.e.,
the square contains a flat piece of the capstone on the top of the stack.
It contains a 0 otherwise.

int road_array[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
};
*/

bool find_road(int (&road_array)[5][5])
{
    // Setup
    int road_array_copy[5][5];
    for (int file = 0; file < 5; file++)
    {
        for (int rank = 0; rank < 5; rank++)
        {
            road_array_copy[file][rank] = road_array[file][rank];
        }
    }
    int dir[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    int row = 5;
    int col = 5;
    queue<pair<int, int>> vertical_q;
    queue<pair<int, int>> horizontal_q;
    bool vertical = false, horizontal = false;

    // Vertical: start searching from the bottom edge
    for (int i = 0; i < 5; i++)
    {
        if (road_array[i][0] == 1)
        {
            vertical_q.push(make_pair(i, 0));
            vertical = true;
        }
    }
    // horizontal: start searching from the left edge
    for (int i = 0; i < 5; i++)
    {
        if (road_array_copy[0][i] == 1)
        {
            horizontal_q.push(make_pair(0, i));
            horizontal = true;
        }
    }

    // Perform vertical search until queue is empty
    while (vertical_q.size() > 0)
    {
        pair<int, int> p = vertical_q.front();
        vertical_q.pop();

        // mark as visited
        road_array[p.first][p.second] = -1;

        // The top of the board is reached - there is a vertical road
        if (p.second == 4)
            return true;

        // check all four directions
        for (int i = 0; i < 4; i++)
        {
            // using the direction array
            int a = p.first + dir[i][0];
            int b = p.second + dir[i][1];

            // not blocked and road square
            if (road_array[a][b] > 0 && a >= 0 && b >= 0 && a < row && b < col)
            {
                vertical_q.push(make_pair(a, b));
            }
        }
    }

    // Perform horizontal search until queue is empty
    while (horizontal_q.size() > 0)
    {
        pair<int, int> p = horizontal_q.front();
        horizontal_q.pop();

        // mark as visited
        road_array_copy[p.first][p.second] = -1;

        // The right edge of the board is reached - there is a horizontal road
        if (p.first == 4)
            return true;

        // check all four directions
        for (int i = 0; i < 4; i++)
        {
            // using the direction array
            int a = p.first + dir[i][0];
            int b = p.second + dir[i][1];

            // not blocked and road square
            // The first conditional makes sure the piece is not blocked or already visited
            if (road_array_copy[a][b] > 0 && a >= 0 && b >= 0 && a < row && b < col)
            {
                horizontal_q.push(make_pair(a, b));
            }
        }
    }

    // default
    return false;
}
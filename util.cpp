#include <string>
#include <stdexcept>
#include <map>
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
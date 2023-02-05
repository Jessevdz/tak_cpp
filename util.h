#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <map>

using namespace std;

/*************************************
Define PTN variables
*************************************/
extern map<string, int> _rank_to_index;
extern map<string, int> _file_to_index;
extern vector<string> _drop_counts_one;
extern vector<string> _drop_counts_two;
extern vector<string> _drop_counts_three;
extern vector<string> _drop_counts_four;
extern vector<string> _drop_counts_five;
extern map<string, vector<string>> _valid_drop_counts;
extern vector<string> _file;
extern vector<string> _rank;
extern vector<string> _stone_type;
extern vector<string> _nr_of_stones;
extern vector<string> _directions;

/*************************************
UTILITY FUNCTIONS FOR RUNNING THE GAME
*************************************/
string player_as_string(const char &);
map<int, string> enumerate_all_ptn_moves();
bool find_road(int (&)[5][5]);

#endif
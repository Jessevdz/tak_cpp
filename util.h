#ifndef UTIL_H
#define UTIL_h

#include <string>
#include <map>

/*************************************
UTILITY FUNCTIONS FOR RUNNING THE GAME
*************************************/

std::string player_as_string(const char &);
std::map<int, std::string> enumerate_all_ptn_moves();
bool find_road(int (&)[5][5]);

#endif
/*******************************************
 * Program: csc456_HW2: Dash
 * Author: Nathan Ducasse
 * Class: CSC 456 Operating Systems
 * Due: March 26, 2018
 * Description: See documentation.
 *
 * Compilation: > make
 * Usage: > ./dash
********************************************/
#ifndef DASH_H_
#define DASH_H_

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <string.h>
#include <vector>
#include <sstream>

#define MAX 100
#define PIGS_FLY false

using namespace std;

string trim(string);
string cmd_cmdnm(string);
string cmd_cmdnm(int, char *[MAX]);

bool cmd_pid(int, char *[MAX]);
bool process_menu(int, char *[MAX]);

void cmd_systat();
void signalHandler(int);
void signalHandlerSIGSEGV(int);
void parse_cmd(int, char *[MAX]);
void pipe_cmd(char *[MAX], char *[MAX]);
void cmd_cd(int num_args, char *args[MAX]);
void redir_cmd_out(char *[MAX], char *[MAX]);
void redir_cmd_in(char *[MAX], char *[MAX]);

int get_input(char *[MAX]);
int check_pipe_redirect(int, char *[MAX], char *[MAX], char *[MAX]);

#endif

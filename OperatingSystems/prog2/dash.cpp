/*******************************************
 * File: dash.cpp
 * Description: Contains the main function
********************************************/
#include "dash.h"

int main() {

    signal(SIGINT, signalHandler);
    signal(SIGABRT, signalHandler);
    signal(SIGFPE, signalHandler);
    signal(SIGILL, signalHandler);
    signal(SIGSEGV, signalHandlerSIGSEGV);
    signal(SIGTERM, signalHandler);

    char *args[MAX]; 
    char *cmd1[MAX];
    char *cmd2[MAX];
    int num_args = 0;
    int arg_pipe;

    while(!PIGS_FLY) {
        printf("dash> "); // Display the prompt
        num_args = get_input(args); // Get input and tokenize

        // Continue if empty line
        if(num_args == 0) {
            continue;
        }

        // Check for exit first
        if(strcmp(args[0], "quit") == 0 || strcmp(args[0], "exit") == 0) {
            return 0;
        }

        // Check for |, <, or >, and split tokens accordingly
        arg_pipe = check_pipe_redirect(num_args, args, cmd1, cmd2);

        // Determine which pipe or redirection the command is, and call
	// the appropriate function
        if(arg_pipe == 0) {
            pipe_cmd(cmd1, cmd2);
        } else if(arg_pipe == 1) {
            redir_cmd_out(cmd1, cmd2);
        } else if(arg_pipe == 2) {
            redir_cmd_in(cmd1, cmd2);
        } else {
            parse_cmd(num_args, args);
        }

	// Clear the args variable
        for(int i = 0; i<num_args; i++) {
            args[i] = NULL;
        }
    }
    return 0;
}

/*******************************************
 * File: functions.cpp
 * Description: Contains all the non-main
 *  functions.
********************************************/
#include "dash.h"

/*******************************************
 * Generic handler for signal reception
********************************************/
void signalHandler( int signum ) {
    if(signum == SIGSEGV) {
        signalHandlerSIGSEGV(signum);
    }
    printf("Interrupt signal (%d) received.\n", signum);

    exit(signum);  
}

/*******************************************
 * Signal handler for seg faults. Makes the
 * program exit cleanly (separated to avoid an
 * unclean exit) following a possible infinite
 * loop after the "quit" command is entered
 * under certain conditions.
********************************************/
void signalHandlerSIGSEGV(int signum) {
    exit(signum);
}

/*******************************************
 * Gets the input for every repl iteration
 * and tokenizes it, split on spaces, and gets
 * the number of arguments, then null terminates
 * the argument list.
********************************************/
int get_input(char *args[MAX]) {
    char *str;
    string in, arg;
    int num_args = 0;

    getline(cin, in);
    in = trim(in);

    if(in.empty()) {
        args[0] = NULL;
        return 0;
    }

    stringstream ss(in);
    while(ss >> arg) {
        str = new char[arg.size()+1];
        strcpy(str, arg.c_str());
        args[num_args] = str;

        num_args++;
    }

    args[num_args] = NULL;
    return num_args;
}

/*******************************************
 * This function checks the args array for
 * the first instance of |, <, or >, and
 * returns 0 if |, 1 if <, 2 if >, or 3 if none.
 * It also separates the args on either side of
 * the operator into cmd1 for the left, and cmd2
 * for the right.
********************************************/
int check_pipe_redirect(int num_args, char *args[MAX], char *cmd1[MAX], char *cmd2[MAX]) {
    int res = 3;
    int pip_idx = -1;

    // Detect the operator
    for(int i = 0; i<num_args; i++) {
        if(strcmp(args[i], "|") == 0) {
            res = 0;
            pip_idx = i;
        } else if(strcmp(args[i], ">") == 0) {
            res = 1;
            pip_idx = i;
        } else if(strcmp(args[i], "<") == 0) {
            res = 2;
            pip_idx = i;
        }
    }

    // Split the args (if applicable)
    if(res != 3) {
        for(int i = 0; i<pip_idx; i++) {
            cmd1[i] = args[i];
        }
        
        int count = 0;
        for(int i = pip_idx + 1; i < num_args; i++) {
            cmd2[count] = args[i];
            count++;
        }

        cmd1[pip_idx] = NULL;
        cmd2[pip_idx] = NULL;
    }
    return res;
}

/*******************************************
 * This function expects a non-redirected/piped
 * input, and checks first if it is one of the
 * predefined functions, then if it is a cd command,
 * then a signal, and finally it assumes it is
 * a generic console command and executes accordingly.
********************************************/
void parse_cmd(int num_args, char *args[MAX]) {

    // Check if predefined command
    if(process_menu(num_args, args)) {
        return;
    }

    // Check if cd command
    if(strcmp(args[0], "cd") == 0) {
        cmd_cd(num_args, args);
        return;
    }

    // Check if signal command
    if(strcmp(args[0], "signal") == 0) {
        if(num_args != 3) {
            kill(atoi(args[2]), atoi(args[1]));
        }
        return;
    }

    // Assume generic console command
    int waitpid;
    int childpid;
    int status;

    childpid = fork(); 
    if (childpid == 0) 
    {
        execvp(args[0], args); 
        perror("Exec failed: ");
        printf("Shell process %d exited with status %d\n", waitpid, (status >> 8));  
        exit(5); 
    }

    waitpid = wait(&status);
}

/*******************************************
 * This function checks if it needs to run the 
 * given command as one of the predefined commands.
 * If it does not execute any, it returns false.
 * If it attempts to execute, it will return
 * true regardless of whether it was successfully
 * executed.
********************************************/
bool process_menu(int num_args, char *args[MAX]) {

    string arg;
    bool pidfound = false;

    // Parse command string
    if(strcmp(args[0], "cmdnm") == 0) {
        if(num_args != 2) {
            return true;
        }

        // cmdnm command finds process name for given pid
        arg = cmd_cmdnm(num_args, args);

        // This handles errors and successful finds!
        // Empty string is returned if not found or an error occurs
        if(arg.length() > 0) {
            cout << "Process name: " << arg << endl;
        } else {
            cout << "Process not found!" << endl;
        }

        return true;

    } else if(strcmp(args[0], "pid") == 0) {
        if(num_args != 2) {
            return true;
        }
        // pid command finds pids for all processes that contain
        // the given string as a substring.
        
        // cmd_pid returns a bool for error checking. It outputs
        // the pid in the function, rather than here.
        if(num_args > 1) {
            pidfound = cmd_pid(num_args, args);
        }
        if(!pidfound) {
            cout << "Could not find PID for given process." << endl;
        }

        return true;

    } else if(strcmp(args[0], "systat") == 0) {
        // Just calls systat for system info. Info is printed
        // in the function.
        cmd_systat();

        return true;

    }

    // Any non-predefined-command just goes here directly
    return false;
}

/*******************************************
 * Executes the cd command. Will work with
 * relative and absolute directory changes,
 * and also changes to the home directory
 * defined in the "HOME" environment variable
 * if there was no path given.
********************************************/
void cmd_cd(int num_args, char *args[MAX]) {
    if(num_args>1) {
        chdir(args[1]);
    } else {
        chdir(getenv("HOME"));
    }
}

/*******************************************
 * Executes a pipe operation. NOTE: Does not
 * chain, either with pipes or redirects, and
 * assumes there is one pipe with atomic left-
 * and right-hand sides.
********************************************/
void pipe_cmd(char *cmd1[MAX], char *cmd2[MAX]) {
    int fd_pipe[2];
    int pid1;
    int pid2;
    int status;
    int wpid;

    // Fork it
    pid1 = fork();
    if (pid1 == 0) {
        // child process executes here for input side of pipe
        pipe(fd_pipe); // create pipe

        // Fork it
        pid2 = fork();
        if (pid2 == 0) {
            // grandchild process executes here for output side of pipe
            close(1);              // close standard output
            dup(fd_pipe[1]);       // redirect the output
            close(fd_pipe[0]);     // close unnecessary file descriptor
            close(fd_pipe[1]);     // close unnecessary file descriptor
            execvp(cmd1[0], cmd1);
            exit(1);
        }
        // back to process for input side of pipe
        close(0);              // close standard input
        dup(fd_pipe[0]);       // redirect the input
        close(fd_pipe[0]);     // close unnecessary file descriptor
        close(fd_pipe[1]);     // close unnecessary file descriptor
        execvp(cmd2[0], cmd2);
        exit(1);
    } else {
        // parent process executes here
        wpid = wait(&status);
    }
}

/*******************************************
 * Executes a redirect out operation. NOTE: Does not
 * chain, either with pipes or redirects, and
 * assumes there is one redirect with atomic left-
 * and right-hand sides.
********************************************/
void redir_cmd_out(char *cmd1[MAX], char *cmd2[MAX]) {
    int pid1 = fork();
    int wpid;
    int status;
    if(pid1 == 0) {
        int out = open(cmd2[0], O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IRGRP | S_IWGRP | S_IWUSR);
        dup2(out, 1);
        close(out);
        execvp(cmd1[0], cmd1);
    }
    wpid = wait(&status);

}

/*******************************************
 * Executes a redirect in operation. NOTE: Does not
 * chain, either with pipes or redirects, and
 * assumes there is one redirect with atomic left-
 * and right-hand sides.
********************************************/
void redir_cmd_in(char *cmd1[MAX], char *cmd2[MAX]) {
    int pid1 = fork();
    int wpid;
    int status;
    if(pid1 == 0) {
        int in = open(cmd2[0], O_RDONLY);
        dup2(in, 0);
        close(in);
        execvp(cmd1[0], cmd1);
    }
    wpid = wait(&status);   
}

/*******************************************
 * Returns the process name of the process
 * with the given pid via a string.
********************************************/
string cmd_cmdnm(int num_args, char *args[MAX]) {
    bool next = false;
    ifstream fin;
    string temp;
    string s(args[1]);
    fin.open("/proc/" + s + "/status");
    if(!fin) return ""; // Error returns empty string
    
    // Search file for "Name" which tells that next fin is the name
    while(fin >> temp) {
        if(next) {
            fin.close();
            return temp; // Return found string
        }
        if(temp.find("Name") != string::npos) {
            next = true;
        }
    }
    fin.close();
    return ""; // Not found returns empty string
}

/*******************************************
 * Same as the other function with the same
 * name, but used as a utility/helper function
 * for cmd_pid
********************************************/
string cmd_cmdnm(string n) {
    bool next = false;
    ifstream fin;
    string temp;
    fin.open("/proc/" + n + "/status", ios::in);
    if(!fin) return ""; // Error returns empty string
    
    // Search file for "Name" which tells that next fin is the name
    while(fin >> temp) {
        if(next) {
            fin.close();
            return temp; // Return found string
        }
        if(temp.find("Name") != string::npos) {
            next = true;
        }
    }
    fin.close();
    return ""; // Not found returns empty string
}

/*******************************************
 * Prints out the pids of the processes whose
 * names contain the given substring. Returns
 * a bool for if there was a process that 
 * contained the substring or not
********************************************/
bool cmd_pid(int num_args, char *args[MAX]) {
    int pid_max, pid = -1;
    ifstream fin;
    bool found = false;
    string temp;
    
    // Find pid_max dynamically
    fin.open("/proc/sys/kernel/pid_max", ios::in);
    
    if(!fin) {
        cout << "Error finding pid_max" << endl;
        fin.close();
        return false;
    }
    
    fin >> temp;
    fin.close();
    pid_max = stoi(temp);
    bool isfirst = true;
    cout << "PIDs: "; 
    // Loop up to find else to pid_max
    while(pid <= pid_max) {
        if(num_args > 1 && cmd_cmdnm(to_string(pid)).find(args[1]) != string::npos) {
            if(isfirst) {
                cout << pid;
                isfirst = false;
            } else {
                cout << ", " << pid;
            }
            found = true;
        }
        pid++;
    }
    cout << endl;
    // true if found, etc.
    return found;
}

/*******************************************
 * Prints out Linux version, system uptime,
 * memory info, and cpu info.
********************************************/
void cmd_systat() {
    cout << endl;
    ifstream fin;
    string temp;
    
    // Get version
    fin.open("/proc/version", ios::in);
    if(!fin) {
        cout << "Error finding linux version" << endl;
    } else {
        getline(fin, temp);
        cout << "--Linux Version Info--" << endl;
        cout << temp << endl;
    }
    cout << endl;
    fin.close();
    
    // Get uptime
    fin.open("/proc/uptime");
    if(!fin) {
        cout << "Error finding linux uptime" << endl;
    } else {
        fin >> temp;
        cout << "--Linux Uptime--\n" << temp << " seconds" << endl;
    }
    cout << endl;
    fin.close();
    
    // Get memory info
    fin.open("/proc/meminfo", ios::in);
    if(!fin) {
        cout << "Error finding memory information" << endl;
    } else {
        cout << "--Memory Information--" << endl;
        while(getline(fin, temp)) {
            cout << temp << endl;
            if(temp.find("MemAvailable") != string::npos) break;
        }
    }
    cout << endl;
    fin.close();
    
    // Get CPU info
    fin.open("/proc/cpuinfo", ios::in);
    if(!fin) {
        cout << "Error getting CPU info" << endl;
    } else {
        cout << "--CPU Info--" << endl;
        bool atvendor = false, atcache = false;
        while(getline(fin, temp) && !atcache) {
            if(temp.find("vendor_id") != string::npos) {
                atvendor = true;
            }
            if(atvendor) {
                cout << temp << endl;
            }
            if(temp.find("cache size") != string::npos) {
                atcache = true;
            }
        }
    }
    
    cout << endl;
    fin.close();
}

/*******************************************
 * Trims strings and removes whitespace
********************************************/
string trim(string str) {
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    if(first == last) {
        return "";
    }
    if (first == string::npos)
    {
        return str;
    }
    return str.substr(first, (last - first + 1));
}


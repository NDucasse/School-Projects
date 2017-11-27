/*======================================================================
    Program:        Parser.java
    Class:          CSC461 Programming Languages
    Instructor:     Dr. Weiss
    Date:           Fall 2017
    Authors:        Ian Beamer, Nathan Ducasse

    Description: FILL ME IN

    Usage: Use the makefile to create the class files, then run as:
            java Parser <-t>
            where -t is optional

    Input: The user is asked to input an expression

    Output: The user's expression in quotes, and a statement as to whether or
            not the expression is valid, and possibly tokens.

======================================================================*/

import java.io.*;
import java.util.*;

public class Parser {
    public static final HashMap<String, String> OPS = new HashMap<String, String>();

    // Populate OPS 
    static {
        OPS.put ( "letter", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_" );
        OPS.put ( "digit", "0123456789" );
        OPS.put ( "addop", "+-" );
        OPS.put ( "mulop", "*/%" );
        OPS.put ( "special", "()." );
    }


    /*======================================================================
    Authors:        Ian Beamer, Nathan Ducasse
    Description:    Main checks for command line arguments, takes in the 
                    expression, and prints the results

    param in: args  Command line arguments, -t is the only accepted one

    ======================================================================*/
    public static void main ( String []args ) {
        boolean flag = false;

        // check args for -t flag
        if ( args.length > 1 ) {
            System.out.println ( "Usage: java Parser <-t>");
        } else if ( args.length == 1 ) {
            if ( !args[0].equals ( "-t" ) ) {
                System.out.println ( "Invalid flag: " + args[0] );
                System.out.println ( "Usage: java Parser <-t>");
                return;
            }
            flag = true;
        }

        // get input
        System.out.print ( "Enter expression: " );
        Scanner sc = new Scanner ( System.in );
        String expr = sc.nextLine();

        // loop until blank input
        while ( !expr.trim().isEmpty() ) {
            if ( !CheckSyntax ( expr ) ) {
                System.out.println ( "\"" + expr +
                                     "\" is not a valid expresion" );
            } else {
                System.out.println ( "\"" + expr +
                                     "\" is a valid expresion" );
            }

            // tokenize function is only called if the flag is set
            if ( flag ) {
                Tokenize ( expr );
            }

            // get next input
            System.out.println();
            System.out.println();
            System.out.print ( "Enter expression: " );
            expr = sc.nextLine();
        }

        System.out.print ( "(end of input)\n" );
        return;
    }


    /*======================================================================
    Authors:        Nathan Ducasse, Ian Beamer
    Description:    Used when -t is found, tokenizes string by parsing
                    each possible entry into the various operations or
                    integers

    param in: in    Input string from Main

    ======================================================================*/
    public static void Tokenize ( String in ) {
        String strsplit[] = in.split ( "" );

        LinkedHashMap<Integer, String> tokens = new LinkedHashMap<Integer, String>();

        String currtoken = "";
        int token = 0;
        int i = 0;

        // loops through the individual characters and tokenizes
        for ( i = 0; i < strsplit.length && strsplit[i] != null 
            && !strsplit[i].isEmpty(); i++ ) {
            // If a space is encountered, that signifies a separate token
            // (if there is a nonempty current token)
            if ( strsplit[i].equals ( " " ) && !currtoken.isEmpty() ) {
                tokens.put ( token, currtoken );
                currtoken = "";
                token++;
                continue;
            }

            // Accounts for multiple whitespace
            if ( strsplit[i].equals ( " " ) ) {
                continue;
            }

            // add any non-letter, non-digit character as its own token
            // (whitespace handled above doesn't reach this point)
            // Only pushes currtoken if it isn't empty.
            if ( currtoken.isEmpty() 
                && ( !OPS.get ( "letter" ).contains ( strsplit[i] ) 
                && !OPS.get ( "digit" ).contains ( strsplit[i] ) ) ) {

                tokens.put ( token, strsplit[i] );
                token++;
            } else if ( !OPS.get ( "letter" ).contains ( strsplit[i] ) 
                && !OPS.get ( "digit" ).contains ( strsplit[i] ) ) {

                tokens.put ( token, currtoken );
                token++;

                tokens.put ( token, strsplit[i] );
                token++;

                currtoken = "";
            // push letters and digits
            } else if ( OPS.get ( "letter" ).contains ( strsplit[i] ) 
                || OPS.get ( "digit" ).contains ( strsplit[i] ) ) {

                currtoken = currtoken + strsplit[i];
            }
        }
        // check if there is a last token at the end of the string
        if ( !currtoken.isEmpty() ) {
            tokens.put ( i, currtoken );
        }

        Iterator it = tokens.entrySet().iterator();
        // print list of tokens
        while ( it.hasNext() ) {
            Map.Entry pair = ( Map.Entry ) it.next();
            System.out.print ( pair.getValue() );
            if ( it.hasNext() ) {
                System.out.print ( ", " );
            } else {
                System.out.println();
            }
        }
    }


    /*======================================================================
    Authors:        Nathan Ducasse, Ian Beamer
    Description:    Checks syntax of the expression, token by token, trying
                    to find inconsistencies.

    param in: tokeninput - Input string from Main (or recursive call)

    output:         False - function found an inconsistency in the expression
                    True - function found no inconsistencies in the expression

    ======================================================================*/
    public static boolean CheckSyntax ( String tokeninput ) {
        tokeninput = tokeninput.trim();
        int numopenparens = 0;
        String []tokenchars = tokeninput.split ( "" );
        String currtoken = "";
        
        // check string for incorrect syntax character by character
        for ( int i = 0; i < tokenchars.length; i++ ) {
            if ( tokenchars[i].isEmpty() ) {
                continue;
            }

            // Count number of '(' and if it's the first one, ignore it
            if ( tokenchars[i].equals ( "(" ) ) {
                if ( numopenparens > 0 ) {
                    currtoken += "(";
                } else {
                    // if it's the first '(', check previous characters for operator
                    for ( int j = i - 1; j >= 0; j-- ) {
                        if ( OPS.get ( "addop" ).contains ( tokenchars[j] ) 
                            || OPS.get ( "mulop" ).contains ( tokenchars[j] ) ) {

                            break;
                        } else if ( !tokenchars[j].equals ( " " ) ) {
                            return false;
                        }
                    }
                }
                numopenparens++;

            // Push ')' to current token as long as it's not the outside closing paren.
            // If it's the last one, either there's an operator after or
            // it's the end of the string. If there's anything else, it's not valid.
            } else if ( tokenchars[i].equals ( ")" ) ) {
                numopenparens--;

                // If not outermost paren, just push it to the curr token.
                if ( numopenparens > 0 ) {
                    currtoken += ")";
                } else if ( numopenparens < 0 ) {
                    // If there's just a random closing paren it isn't valid.
                    return false;
                } else {

                    // Check the characters after the last paren.
                    // If it finds an operator, it returns the validity of the
                    // tokens before and after the operator.
                    if ( i == tokenchars.length - 1 ) {
                        return CheckSyntax(currtoken);
                    }

                    for ( int j = i + 1; j < tokenchars.length; j++ ) {
                        if ( OPS.get ( "addop" ).contains ( tokenchars[j] ) 
                            || OPS.get ( "mulop" ).contains ( tokenchars[j] ) 
                            || tokenchars[j].trim().isEmpty() ) {
                            // operator is found, check validity of operation
                            return CheckSyntax ( currtoken ) &&
                                   CheckSyntax ( tokeninput.substring ( j + 1 ) );

                        } else if ( !tokenchars[j].equals ( " " ) ) {
                            return false;
                        }
                    }

                    // If it gets to the end without finding a non-whitespace char,
                    // return the validity of the current token (ignoring outermost parens).
                    return CheckSyntax ( currtoken );
                }

            // check for '*', '/', '%', or '+'
            } else if ( OPS.get ( "mulop" ).contains ( tokenchars[i] ) 
                || tokenchars[i].equals ( "+" ) ) {
                
                // check for overarching parentheses
                if ( numopenparens > 0 ) {
                    currtoken += tokenchars[i];
                } else {
                    // both sides of expression must be valid
                    return CheckSyntax ( tokeninput.substring ( 0, i ) ) 
                        && CheckSyntax ( tokeninput.substring ( i + 1, tokeninput.length() ) );
                }

            // '-' has two operations so it is handled separately
            } else if ( tokenchars[i].equals ( "-" ) ) {
                // check for overarching parentheses
                if ( numopenparens > 0 ) {
                    currtoken += tokenchars[i];
                } else {
                    // If left side is invalid, check if the previous nonwhitespace
                    // character was an operator or '('. If it was, check right side
                    // for validity, else it was invalid.
                    if(!CheckSyntax ( tokeninput.substring ( 0, i ) ) ) {
                        for ( int j = i - 1; j >= 0; j-- ) {
                            if ( !tokenchars[j].equals ( " " )
                                && !OPS.get ( "letter" ).contains ( tokenchars[j] ) 
                                && !OPS.get ( "digit" ).contains ( tokenchars[j] ) 
                                && !tokenchars[j].equals ( ")" )
                                && !tokenchars[j].equals ( "." ) ) {

                                break;
                            } else if ( !tokenchars[j].equals ( " " ) ) {
                                return false;
                            }
                        }
                        return CheckSyntax ( tokeninput.substring ( i+1, tokeninput.length() ) );
                    }
                    // If the left side was valid, check both sides.
                    return CheckSyntax ( tokeninput.substring ( 0, i ) ) 
                        && CheckSyntax ( tokeninput.substring ( i+1, tokeninput.length() ) );
                }
            // Anything else gets added to the current token.
            } else {
                currtoken += tokenchars[i];
            }
        }
        // base case, check the whole input for basic syntax
        return CheckOperandSyntax ( tokeninput );
    }


    /*======================================================================
    Authors:        Nathan Ducasse, Ian Beamer
    Description:    Checks syntax validity of the given base token

    param in: in    Input string from CheckSyntax

    output:         False - function found an inconsistency in the expression
                    True - function found no inconsistencies in the expression

    ======================================================================*/
    public static boolean CheckOperandSyntax ( String currtoken ) {
        currtoken = currtoken.trim();
        String []strsplit = currtoken.split ( "" );

        // empty tokens are not valid (for case "()" )
        if ( currtoken.isEmpty() ) {
            return false;
        }

        // Check if floating point
        if ( currtoken.contains ( "." ) ) {
            return CheckFloatSyntax ( currtoken );
        }

        // Check if integer
        if ( OPS.get ( "digit" ).contains ( strsplit[0] ) ) {
            return CheckIntSyntax ( currtoken );
        }

        // Check for invalid non-letter, non-digit characters
        // Whitespace here is considered invalid.
        for ( int i = 0; i < strsplit.length; i++ ) {
            if ( !OPS.get ( "letter" ).contains ( strsplit[i] ) 
                && !OPS.get ( "digit" ).contains ( strsplit[i] ) ) {
                return false;
            }
        }
        // If it survives the gauntlet, it's valid
        return true;
    }


    /*======================================================================
    Authors:        Nathan Ducasse, Ian Beamer
    Description:    Checks if given token is a floating point.

    param in: in    Input string from CheckSyntax

    output:         False - function found an inconsistency in the expression
                    True - function found no inconsistencies in the expression

    ======================================================================*/
    public static boolean CheckFloatSyntax ( String currtoken ) {
        currtoken = currtoken.trim();
        String []strsplit = currtoken.split ( "" );
        boolean decimal = false;

        // decimal point at start or end of token is invalid
        if ( strsplit[0].equals ( "." ) 
            || strsplit[strsplit.length - 1].equals ( "." ) ) {

            return false;
        }

        // any non-digit, non-period character is invalid
        for ( int i = 0; i < strsplit.length; i++ ) {
            if ( !OPS.get ( "digit" ).contains ( strsplit[i] ) &&
                 !strsplit[i].equals ( "." ) ) {
                return false;
            }

            // Check for more than one decimal point
            if ( strsplit[i].equals ( "." ) ) {
                if ( decimal ) {
                    return false;
                }
                decimal = true;
            }
        }
        // If it gets here, it's valid
        return true;
    }


    /*======================================================================
    Authors:        Nathan Ducasse, Ian Beamer
    Description:    Checks if given token is a valid integer

    param in: in    Input string from CheckSyntax

    output:         False - function found an inconsistency in the expression
                    True - function found no inconsistencies in the expression

    ======================================================================*/
    public static boolean CheckIntSyntax ( String currtoken ) {
        currtoken = currtoken.trim();
        String []strsplit = currtoken.split ( "" );

        // Check for non-digit characters
        for ( int i = 0; i < strsplit.length; i++ ) {
            if ( !OPS.get ( "digit" ).contains ( strsplit[i] ) ) {
                return false;
            }
        }
        return true;
    }

}

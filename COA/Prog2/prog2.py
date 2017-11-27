import sys

CONST_OPCODES = (
    ('HALT', 'LD', 'ADD', 'J'),
    ('NOP', 'ST', 'SUB', 'JZ'),
    ('?', 'EM', 'CLR', 'JN'),
    ('?', '?', 'COM', 'JP'),
    ('?', '?', 'AND', '?'),
    ('?', '?', 'OR', '?'),
    ('?', '?', 'XOR', '?'),
    ('?', '?', '?', '?'),
    ('?', 'LDX', 'ADDX', '?'),
    ('?', 'STX', 'SUBX', '?'),
    ('?', 'EMX', 'CLRX', '?'),
    ('?', '?', '?', '?'),
    ('?', '?', '?', '?'),
    ('?', '?', '?', '?'),
    ('?', '?', '?', '?'),
    ('?', '?', '?', '?')
)


# Main function
def main():
    # Mem is the emulator memory, pc is the program counter.
    mem = [0] * 4096
    pc = 0
    memdumpflag = False
    parseflag = False

    if len(sys.argv) <= 1:
        return

    # Check for flags
    if len(sys.argv) == 3:
        if sys.argv[2] == '-memdump':
            memdumpflag = True
        elif sys.argv[2] == '-parse':
            parseflag = True
        else:
            print ('Improper flags.')
            return
    elif len(sys.argv) == 4:
        if (sys.argv[2] == '-memdump' and sys.argv[3] == '-parse') or (sys.argv[3] == '-memdump' and sys.argv[2] == '-parse'):
            memdumpflag = True
            parseflag = True
        else:
            print ('Improper flags.')
            return
    
    # Get file input and separate into multiple lines
    filein = open(sys.argv[1])
    obfile = filein.read()
    obfile = obfile.split('\n')[0:-1]

    for line in obfile:
        # Split the line into different arguments
        line = line.split(' ')
        # If the line only has one argument, it's the pc.
        if len(line) == 1:
            pc = int(line[0], 16)
        elif len(line) >1:
            # Input the data into memory
            for i in range(int(line[1])):
                # I found that lists can be accessed in any base
                # Interpret line[0] in hex then add i to get the next memory position.
                mem[int(line[0], 16) + int(hex(i), 16)] = int(line[2+i], 16)

    do_instructions(mem, pc)
    # Call parse and memdump if flags are specified
    if parseflag:
        parse(mem)
    if memdumpflag:
        memdump(mem)
   

# Executes the object file
def do_instructions(mem, pc):
    # create registers and flags
    ir = 0
    ac = 0
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    halt = False
    unimp = False
    unimpop = False
    invalid = False
    illegal = False
    
    # Loop through valid instructions in memory
    while not halt and pc < len(mem):
        # Update current instruction and get opcode, etc.
        ir = mem[pc]
        branch = False
        am = (ir >> 2) & 0b1111
        oc = (ir >> 6) & 0b111111
        addr = ir >> 12
        opname = CONST_OPCODES[oc & 0b1111][oc >> 4] or '.'
        
        print ('{0:03x}:  {1:06x}  '.format(pc, ir), end='')

        # Check for opcode validity
        if not is_valid_op(opname):
            print ('????  ', end='')
            halt = True
            invalid = True

        else:
            if opname in ('LDX', 'STX', 'EMX', 'ADDX', 'SUBX', 'CLRX'):
                unimpop = True
                halt = True
            print ('{:6}'.format(opname), end='')    

        # Do AM-impartial operations
        if opname == 'HALT':
            print ('     ', end='')
            halt = True

        elif opname == 'NOP':
            pass

        elif opname == 'CLR':
            ac = 0

        elif opname == 'COM':
            ac = ~ac

        if am == 0:
            flag = True
            # Check for all the non-branch, non-LD/ST instructions
            if opname in ('ADD', 'SUB', 'AND', 'OR', 'XOR'):
                ac = alu_op(ac, mem[addr], oc)

            # Check for all the memory operations
            elif opname == 'LD':
                ac = mem[addr]
            elif opname == 'ST':
                mem[addr] = ac
            elif opname == 'EM':
                mem[addr] = mem[addr] ^ ac
                ac = mem[addr] ^ ac
                mem[addr] = mem[addr] ^ ac

            # Check for all the branch operations
            elif opname == 'J':
                pc = alu_op(ac, addr, oc)
                branch = True
            elif opname == 'JZ':
                if ac == 0:
                    pc = mem[addr]
                    branch = True
            elif opname == 'JN':
                if ac < 0:
                    pc = mem[addr]
                    branch = True
            elif opname == 'JP':
                if ac > 0:
                    pc = mem[addr]
                    branch = True

            # Check for non-AM-impartial operations
            elif opname not in ('HALT', 'NOP', 'CLR', 'COM'):
                flag = False
                print ('???  ', end='')
                halt = True

            # Don't want to print anything if HALT.
            if flag and not opname == 'HALT':
                print ('{0:03x}  '.format(addr), end='')

        elif am == 1:
            if opname in ('ADD', 'SUB', 'AND', 'OR', 'XOR', 'LD'):
                print ('IMM  ', end='')
                ac = alu_op(ac, addr, oc)

            elif opname in ('.', '?', 'HALT', 'NOP', 'CLR', 'COM', 'LD'):
                # Checks for operations that don't care about AM
                print ('IMM  ', end='')

            else:
                #illegal mode
                print ('???  ', end='')
                illegal = True
                halt = True

        elif am == 2:
            halt = True
            print ('???  ', end='')

            if opname in ('LDX', 'STX', 'EMX', 'ADDX', 'SUBX'):
                #illegal mode
                illegal = True

            else:
                #unimplemented
                unimp = True

        elif am == 3:
            halt = True
            print ('???  ', end='')

            if opname in ('LDX', 'STX', 'EMX', 'ADDX', 'SUBX'):
                #illegal mode
                illegal = True

            else:
                #unimplemented
                unimp = True

        elif am == 4:
            print ('???  ', end='')

            if opname in ('LDX', 'STX', 'EMX', 'ADDX', 'SUBX'):
                #illegal mode
                illegal = True

            else:
                #unimplemented
                unimp = True

        else:
            print ('???  ', end='')
            #illegal mode
            halt = True
            illegal = True

        # Checks for negative values in accumulator and "fixes" them.
        if ac >= 0:
            print ('AC[' + '{0:06x}'.format(ac) + ']  X0[{0:03x}]  X1[{1:03x}]  X2[{2:03x}]  X3[{3:03x}]'.format(x0, x1, x2, x3))

        else:
            print ('AC[' + '{0:06x}'.format(0xffffff + ac+1) + ']  X0[{0:03x}]  X1[{1:03x}]  X2[{2:03x}]  X3[{3:03x}]'.format(x0, x1, x2, x3))

        if not branch:
            pc += 1
        
    # Prints exit code based on error flags
    print ('Machine Halted - ', end='')

    if invalid:
        print ('undefined opcode')

    elif unimpop:
        print ('unimplemented opcode')

    elif illegal:
        print ('illegal addressing mode')

    elif unimp:
        print ('unimplemented addressing mode')

    elif halt:
        print ('HALT instruction executed')
    else:
        print ('Error')


# Checks for valid op name
def is_valid_op(opname):
    if opname in ('.', '?'):
        return False
    return True


# Emulates the ALU
def alu_op(arg1, arg2, op):
    if op >> 4 == 1 or op >> 4 == 3:
        return arg2
    op = op & 0b1111
    if op == 0:
        return arg1 + arg2
    elif op == 1:
        return arg1 - arg2
    elif op == 4:
        return arg1 & arg2
    elif op == 5:
        return arg1 | arg2
    elif op == 6:
        return arg1 ^ arg2
    return arg1
        

# Dumps the (nonzero) contents of mem to output
def memdump(mem):
    print('Memory Dump:')
    for i in range(len(mem)):
        if mem[i] != 0:
            print('{0:03x}:  {1:06x}'.format(i, mem[i]))


# Parse the given address and print it
def parse(mem):
    print('Parsing: Enter address and interval to dump:')
    addr = input('(\'0xfff i\' or \'fff i\'):')
    addr, interval = addr.split()
    # Fix for formatting
    if '0x' in addr:
        addr = addr[2:]
    # Interpret address as hex
    addr = int(addr, 16)
    print('            ADDR       OP     AM')
    # Print out each memory location from the start to the start + interval
    for i in range(int(interval)):
        # Messy, messy format string. This was as awful to code as it is to read.
        print('{:03x}'.format(addr + int(hex(i), 16)) + ':    {0:012b} {1:06b} {2:06b}'.format(int(bin(mem[addr + int(hex(i), 16)]>>12), 2), int(bin(mem[addr + int(hex(i), 16)]>>6)[-6:], 2), int(bin(mem[addr + int(hex(i), 16)])[-6:], 2)))
    print()

# Spoof main
if __name__ == '__main__':
    main()


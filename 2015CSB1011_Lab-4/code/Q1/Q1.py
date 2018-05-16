import sys
import copy
import time

start = time.time()

# Defining queue data structure for using in BFS search

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

# Defining stack data structure for using in Goal Stack Planning (GSP)

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0


file_name = sys.argv

# Initializing expanded nodes to 0
num_nodes_expanded = 0

# Opening input file for reading input
file = open(file_name[1], "r")
line = file.readline()

# Finding number of rooms and boxes
N = int(line[0]) # Number of Rooms
B = int(line[2]) # Number of Boxes

# Finding Algortithm to use
line = file.readline()
line = line.strip()
algo = line # BFS - f GSP - g

# Reading Initial State
line = file.readline() # Discarding word 'initial'
line = file.readline()
line = line.strip()
Init_State = line

# Reading Goal State
line = file.readline() # Discarding word 'goal'
line = file.readline()
line = line.strip()
Goal_State = line

"""
STATE DESCRIPTION

All the states have been stored in the form of a boolean array of size N*B + 2N + B + 1
The first N*B entries denote if any of the predicates at(b1,r1), at(b1,r2),....., at(bm,rn) are true in that state
The next  N   entries denote if any of the predicates at(shakey,r1), at(shakey,r2),...., at(shakey,rn) are true in that state
The next  B   entries denote if any of the predicates on(shakey,b1), on(shakey,b2),...., on(shakey,bm) are true in that state
The next entry denotes if the predicate on(shakey,floor) is true in that state
The last  N   entries denote if any of the predicates swithon(l1), switchon(l2),...., switchon(ln) are true in that state or not 
"""

# Size of State Array
State_Size = (N*B) + (2*N) + B + 1

# Formulating Initial State
temp = Init_State.split(") (")
temp[0] = temp[0][1:] # Ignoring '('
temp[len(temp)-1] = temp[len(temp)-1][:-1] # Ignoring ')'

Init_State = temp

# Boolean Array to store Start State Information
St_State = []
St_State = [0] * State_Size

# Populating the starting state array
for i in range(len(Init_State)):                       
    pos = Init_State[i]
    pos = pos.split(" ")
    
    # at proposition
    if(pos[0] == "at"):
        # Position of a box
        if(pos[1][0] == 'b'):
            num1 = int(pos[1][1]) - 1
            num2 = int(pos[2][1]) - 1
            St_State[num1*N+num2] = 1
        # Position of shakey
        elif(pos[1] == "shakey"):
            num1 = int(pos[2][1]) - 1
            St_State[N*B + num1] = 1

    # on proposition
    elif(pos[0]=="on"):
        # Shakey on box
        if(pos[2] != "floor"):
            num1 = int(pos[2][1]) - 1
            St_State[N*B + N +num1] = 1
        # Shakey on floor
        else:
            St_State[N*B + N + B] = 1
    
    # switchon proposition
    elif(pos[0] == "switchon"):
            num1 = int(pos[1][1]) - 1
            St_State[N*B + N + B + 1 + num1] = 1

# Formulating Goal State
temp = Goal_State.split(") (")
temp[0] = temp[0][1:] # Ignoring '('
temp[len(temp)-1] = temp[len(temp)-1][:-1] # Ignoring ')'

Goal_State = temp

# Boolean Array to store Goal State Information
G_State = []
G_State = [0] * State_Size

# Populating the goal state array
for i in range(len(Goal_State)):                       
    pos = Goal_State[i]
    pos = pos.split(" ")
    
    # at proposition
    if(pos[0] == "at"):
        # Position of a box
        if(pos[1][0] == 'b'):
            num1 = int(pos[1][1]) - 1
            num2 = int(pos[2][1]) - 1
            G_State[num1*N+num2] = 1
        # Position of shakey
        elif(pos[1] == "shakey"):
            num1 = int(pos[2][1]) - 1
            G_State[N*B + num1] = 1

    # on proposition
    elif(pos[0]=="on"):
        # Shakey on box
        if(pos[2] != "floor"):
            num1 = int(pos[2][1]) - 1
            G_State[N*B + N +num1] = 1
        # Shakey on floor
        else:
            G_State[N*B + N + B] = 1
    
    # switchon proposition
    elif(pos[0] == "switchon"):
            num1 = int(pos[1][1]) - 1
            G_State[N*B + N + B + 1 + num1] = 1

"""
Function to find out the successors of a state and return them as a list along with the action required to reach them
GIves only the valid actions by matching the preconditions of all the actions
"""
def Get_Successors(State):            
    
    actions=[]
    # Action - Go (x,y)

    # Pre-conditions - At(shakey,x), On(shakey, floor) 
    if(State[N*B + N + B] == 1): # if shakey is on floor
        for i in range(N*B, N*B + N):
            if(State[i] == 1):
                x = i - N*B + 1
                for j in range(N*B, N*B + N):
                    if(State[j] == 0):
                        y = j - N*B + 1         
                        action = []
                        New_State = copy.deepcopy(State)
                        New_State[i] = 0
                        New_State[j] = 1
                        string = "(go r" + str(x) + " r" + str(y) + ")"     
                        action.append(New_State)             
                        action.append(string)
                        actions.append(action) # Adding go action to the list of valid actions

     # Action - Push (b,x,y)
     # Action - Climbup (b,x)

     # Pre-conditions - At(b,x), At(shakey,x), On(shakey,floor)   
        for i in range(N*B, N*B + N):
            if(State[i] == 1):
                x = i - (N*B) + 1  # room in which shakey is
                start = x
                for j in range(0, B):
                    if(State[(x-1) + j*N] == 1):
                        for k in range(0, N):
                            if(k != x-1):
                                action = []
                                New_State1 = copy.deepcopy(State)
                                New_State1[(x-1) + j*N] = 0
                                New_State1[i] = 0
                                New_State1[k + j*N] = 1
                                New_State1[k + N*B] = 1
                                string1 = "(push b" + str(j+1) + " r" + str(x) + " r" + str(k+1) + ")"
                                action.append(New_State1)
                                action.append(string1)
                                actions.append(action) # Adding push action to the list of valid actions

                        action = []
                        New_State2 = copy.deepcopy(State)
                        New_State2[N*B + N + B] = 0
                        New_State2[N*B + N + j] = 1
                        string2 = "(climbup b" + str(j+1) + " r" + str(x) + ")"
                        action.append(New_State2)
                        action.append(string2)
                        actions.append(action) # Adding climbup action to the list of valid actions
                                    
    # Action - Climbdown(b)

    # Pre-conditions - On(shakey,b)

    else: # if shakey is not on floor
                                                                                   
        for i in range(N*B + N, N*B + N + B): # find the box on which shakey is
            if(State[i] == 1):
                action = []
                x = i - (N*B + N) + 1
                New_State = copy.deepcopy(State)
                New_State[i] = 0
                New_State[N*B + N + B] = 1
                string = "(climbdown b" + str(x) + ")"     
                action.append(New_State)             
                action.append(string)
                actions.append(action) # Adding climbdown action to the list of valid actions

    # Action - Turnon(l,x)

    # Pre-conditions - At(b,x), At(shakey,x), On(shakey,b) ~Switchon(l)
        for i in range(N*B, N*B + N):
            if(State[i] == 1 and State[i + N + B + 1] == 0):
                x = i - (N*B)  # room in which shakey is
                start = x
                for j in range(0, B):
                    if(State[x + j*N] == 1):
                        action = []
                        New_State = copy.deepcopy(State)
                        New_State[i + N + B + 1] = 1
                        string = "(turnon l" + str(x+1) + " r" + str(x+1) + ")"
                        action.append(New_State)
                        action.append(string)
                        actions.append(action) # Adding climbup action to the list of valid actions
            
    
    # Action - Turnoff(l,x)

    # Pre-conditions - At(b,x), At(shakey,x), On(shakey,b) ~Switchon(l)            
        for i in range(N*B, N*B + N):
            if(State[i] == 1 and State[i + N + B + 1] == 1):
                x = i - (N*B)  # room in which shakey is
                start = x
                for j in range(0, B):
                    if(State[x + j*N] == 1):
                        action = []
                        New_State = copy.deepcopy(State)
                        New_State[i + N + B + 1] = 0
                        string = "(turnoff l" + str(x+1) + " r" + str(x+1) + ")"
                        action.append(New_State)
                        action.append(string)
                        actions.append(action) # Adding climbup action to the list of valid actions
        
    return actions

"""
Function which returns true if the current state is a goal state(given in the file) 
or else returns false
"""
def isGoalState(State):
    if(len(State)!=len(G_State)):
        return 0
    for i in range(len(G_State)):
        if(G_State[i] == 1 and State[i] == 0):
            return 0
    return 1

"""
actions = Get_Successors(St_State)

for i in range(len(actions)):
    print(actions[i][0])
    print(actions[i][1])
"""

# BFS search

def BreadthFirstSearch():

    global num_nodes_expanded    # for counting the total number of nodes expanded in the search
    
    """
    Search the shallowest nodes in the search tree first.
    """

    """
    Initializing the queue and the frontier and explored lists
    Also initializing the list of actions to reach the goal state
    """
    queue = Queue() # using queue for bfs because of FIFO priciple     
    State = St_State
    rev_list_of_actions = []
    
    parents = {} # Stores the parent co-ordinates of a given co-ordinate ( key: co-ordinate , value: parent co-ordinate )
    info = {}    # Stores the information of a given co-ordinate in the form of the action taken on its parent co-ordinate to reach it and the 
                 # length of edge traversed to reach it ( key: co-ordinate , value: (co-ordinate,action,edgelength) ) 
        
    explored = {} # Explored list
    frontier = {} # Frontier List
    State = tuple(State)
    parents[State] = None # Initializing for the parent node
    info[State] = None    # Initializing for the parent node
    
    """
    The queue and the frontier list contain the same elements(always)
    Since we cannot check if an element is in the queue or not
    Thats why we are using an addiotional dictionary(frontier list)
    """
    # Pushing the starting state onto the queue and frontier list
    frontier[State] = State 
    queue.push(State)
    
    while(not queue.isEmpty()):
        num_nodes_expanded += 1
        ele = queue.pop()
        del frontier[ele]
        explored[ele] = ele
        State = ele
        
        if(isGoalState(State)): # is goal state
            cur = ele
            while(info[cur] != None):
                rev_list_of_actions.append(info[cur][1]) # adding the actions in the reverse order
                cur = parents[cur]
            list_of_actions=[]
            while(rev_list_of_actions != []): # Reversing the list of actions
                list_of_actions.append(rev_list_of_actions[-1])
                rev_list_of_actions.pop()
            return list_of_actions
        
        suc = Get_Successors(list(State)) 
             
        while(suc != []): #  For all successors
            if tuple(suc[-1][0]) in explored.keys(): # if already explored  
                suc.pop()
            else:
                if tuple(suc[-1][0]) in frontier.keys(): # if in the frontier list
                    suc.pop()
                else:
                    tup = tuple(suc[-1][0]) # else saving the information of the co-ordinate and pushing it on the frontier list
                    parents[tup] = ele
                    frontier[tup] = tup
                    info[tup] = suc[-1]
                    queue.push(tup)
                    suc.pop()
                                
    return [] # If no solution found

"""
actions = BreadthFirstSearch() 
print ("BFS search used")
print ("Solution Length: ",len(actions))
for i in range(len(actions)):
        print (actions[i])
print ("Number of nodes expanded: ",num_nodes_expanded)
"""

# GoalStackPlanning search

def GoalStackPlanning():
    actions_to_take = [] # List of actions to take (initially empty)
    stack = Stack() # Stack initialized
    new = copy.deepcopy(G_State)
    stack.push(new) # pushing the goal state onto the stack
    cur_state = copy.deepcopy(St_State)
    p_cond = [0]*(State_Size)
    
    # Pushing the predicates of the goal state individually onto the stack
    for i in range(State_Size): 
        if(new[i] == 1):
            stack.push(i+1)

    # While stack not empty
    while(not stack.isEmpty()):
        ele = stack.pop()

        # if popped element is a conjunct goal
        if(type(ele) == list):    
            passed = 1
            # Checking if all the predicates are individually true
            for i in range(State_Size):     
                if(ele[i] == 1 and cur_state[i] == 0):
                    passed = 0
                    break
            
            """
            If the predicates are not true then pushing the conjunct goal
            and the individual predicates on the stack again
            """
            if(passed == 0):                                     
                stack.push(ele)
                for i in range(State_Size):
                    if(ele[i] == 1):
                        stack.push(i+1)
        
        # if popped element is a predicate
        elif(type(ele) == int):
            
            # if current state does not satisfy predicate
            if(ele > 0 and cur_state[ele-1] == 0):       
                act = []
                ano = ele - 1
                
                # if predicate is At(b,x), selecting the relevant action to push on the stack
                if(ano < N*B): 
                    b = int((ano)/N)
                    r = int((ano)%N)

                    # Finding room in which box currently is
                    for k in range(0, N):
                        if(cur_state[b*N + k] == 1):
                            r2 = k
                            break

                    # Action - push(b,x,y)
                    act = "(push b" + str(b+1) + " r" + str(r2+1) + " r" + str(r+1) + ")" 
                    stack.push(act)
                    precond = [0]*(State_Size)
                    precond[N*b + r2] = 1 # At(b,x)
                    precond[N*B + r2] = 1 # At(shakey,x)
                    precond[N*B + N + B] = 1 # On(shakey,floor)
                    # Pushing Pre-Condition
                    stack.push(precond)
                    # Pushing Individual Predicates
                    stack.push(N*b + r2 + 1)
                    stack.push(N*B + r2 + 1)
                    stack.push(N*B + N + B + 1)

                # if prdicate is At(shakey, x), selecting the relevant action to push on the stack
                elif(ano < N*B + N):
                    y = ano - (N*B)
                    # Finding room in which shakey currently is
                    for k in range(0,N):
                        if(cur_state[N*B + k] == 1):
                            x = k
                            break

                    # Action - go(x,y)         
                    act = "(go r" + str(x+1) + " r" + str(y+1) + ")"
                    stack.push(act)
                    precond = [0]*(State_Size)
                    precond[N*B + x] = 1 # At(shakey,x)
                    precond[N*B + N + B] = 1 # On(shakey,floor)
                    # Pushing Pre-Condition
                    stack.push(precond)
                    # Pushing Individual Predicates
                    stack.push(N*B + x + 1)
                    stack.push(N*B + N + B + 1)

                # if predicate is On(shakey, b), selecting the relevant action to push on the stack
                elif(ano < N*B + N + B):
                    b = ano - (N*B + N)
                    
                    # Finding room in which the box is present
                    for k in range(0,N):
                        if(cur_state[b*N + k] == 1):
                            x = k
                            break

                    # Action - climbup(b,x)         
                    act = "(climbup b" + str(b+1) + " r" + str(x+1) + ")"
                    stack.push(act)
                    precond = [0]*(State_Size)
                    precond[N*b + x] = 1 # At(b,x)
                    precond[N*B + x] = 1 # At(shakey,x)
                    precond[N*B + N + B] = 1 # On(shakey,floor)
                    # Pushing Pre-Condition
                    stack.push(precond)
                    # Pushing Individual Predicates
                    stack.push(N*b + x + 1)
                    stack.push(N*B + x + 1)
                    stack.push(N*B + N + B + 1)

                # if predicate is On(shakey,floor), selecting the relevant action to push on the stack    
                elif(ano == N*B + N + B):  
                    # Finding box on top of which shakey is present
                    for k in range(0,B):
                        if(cur_state[N*B + N + k] == 1):
                            b = k
                            break

                    # Action - climbdown(b)         
                    act = "(climbdown b" + str(b+1) + ")"
                    stack.push(act)
                    precond = [0]*(State_Size)
                    precond[N*B + N + b] = 1 # On(shakey,b)
                    # Pushing Pre-Condition
                    stack.push(precond)
                    # Pushing Individual Predicates
                    stack.push(N*B + N + b + 1)
                
                # if predicate is Switchon(l), selecting the relevant action to push on the stack        
                elif(ano < N*B + N + B + 1 + N):
                    
                    l = ano - (N*B + N + B + 1)
                    r = l

                    # Finding box in that room if present
                    b = -1
                    for k in range(0, B):
                        if(cur_state[k*N + r] == 1):
                            b = k
                            break
                    # If no box is present in that room
                    if(b == -1):
                        # Finding room in which shakey is present
                        for k in range(0,N):
                            if(cur_state[N*B + k] == 1):
                                s_room = k
                                break

                        # Checking if that room has a box
                        for k in range(0,B):
                            if(cur_state[k*N + s_room] == 1):
                                b = k
                                break

                        if(b == -1):
                            b = 0

                    # Action - turnon(l,r)
                    act = "(turnon l" + str(l+1) + " r" + str(r+1) + ")"
                    stack.push(act)
                    precond = [0]*(State_Size)
                    precond[N*B + N + b] = 1 # On(shakey,b)
                    precond[N*B + r] = 1 # At(shakey,x)
                    precond[b*N + r] = 1 # At(b,x)
                    precond[N*B + N + B + 1 + l] = 0 # ~switchon(l)

                    # Pushing Pre-Condition
                    stack.push(precond)
                    # Pushing Individual Predicates
                    stack.push(N*B + N + b + 1)
                    stack.push(N*B + r + 1)
                    stack.push(N*b + r + 1)
                    stack.push((N*B + N + B + 1 + l + 1) * -1)

            # if predicate is ~Switchon(l), selecting the relevant action to push on the stack
            
            elif(ele < 0 and cur_state[(ele * (-1)) - 1] == 1):
                print("element = ",(ele * (-1)) - 1);
                ano = (ele * (-1)) - 1
                l = ano - (N*B + N + B + 1)
                r = l

                # Finding box in that room if present
                b = -1
                for k in range(0, B):
                    if(cur_state[k*N + r] == 1):
                        b = k
                        break
                # If no box is present in that room
                if(b == -1):
                    # Finding room in which shakey is present
                    for k in range(0,N):
                        if(cur_state[N*B + k] == 1):
                            s_room = k
                            break

                    # Checking if that room has a box
                    for k in range(0,B):
                        if(cur_state[k*N + s_room] == 1):
                            b = k
                            break

                    if(b == -1):
                        b = 0

                # Action - turnoff(l,r)
                act = "(turnoff l" + str(l+1) + " r" + str(r+1) + ")"
                stack.push(act)
                precond = [0]*(State_Size)
                precond[N*B + N + b] = 1 # On(shakey,b)
                precond[N*B + r] = 1 # At(shakey,x)
                precond[b*N + r] = 1 # At(b,x)
                precond[N*B + N + B + 1 + l] = 1 # switchon(l)

                # Pushing Pre-Condition
                stack.push(precond)
                # Pushing Individual Predicates
                stack.push(N*B + N + b + 1)
                stack.push(N*B + r + 1)
                stack.push(N*b + r + 1)
                stack.push(N*B + N + B + 1 + l + 1)


        # if popped element is an action, then applying the action to the current state and storing it
        elif(type(ele) == str):
            actions_to_take.append(ele)
            elements = ele.split(" ")
            
            if(elements[0] == "(go"):
                x = int(elements[1][1])
                y = int(elements[2][1])
                cur_state[N*B + x - 1] = 0       # Not at(shakey,x)
                cur_state[N*B + y - 1] = 1       # at(shakey,y)
            
            elif(elements[0] == "(push"):
                b = int(elements[1][1])
                x = int(elements[2][1])
                y = int(elements[3][1])
                cur_state[N*B + x - 1] = 0       # Not at(shakey,x)
                cur_state[N*B + y - 1] = 1       # at(shakey,y)
                cur_state[N*(b-1) + (x-1)] = 0     # Not at(b,x)
                cur_state[N*(b-1) + (y-1)] = 1     # at(b,y)
                
            elif(elements[0] == "(climbup"):
                b = int(elements[1][1])
                x = int(elements[2][1])
                cur_state[N*B + N + b-1] = 1     # on(shakey,b)
                cur_state[N*B + N + B] = 0       # Not on(shakey,floor)
                
            elif(elements[0] == "(climbdown"):
                b = int(elements[1][1])
                cur_state[N*B + N + b-1] = 0     # Not on(shakey,b)
                cur_state[N*B + N + B] = 1       # on(shakey,floor)
                
            elif(elements[0] == "(turnon"):
                l = int(elements[1][1])
                x = int(elements[2][1])
                cur_state[N*B + N + B + 1 + l-1] = 1     # switchon(l)
                
            elif(elements[0] == "(turnoff"):
                l = int(elements[1][1])
                x = int(elements[2][1])
                cur_state[N*B + N + B + 1 + l-1] = 0     # ~switchon(l)

    # returning the final list of actions
    return actions_to_take

############## main ##############

file.close()        

file = open("output_" + file_name[1], "w")

# If algo = BFS
if(algo == "f"):                            
    actions = BreadthFirstSearch() 
    print ("BFS search used")
    print ("Solution Length: ", len(actions))
    file.write(str(len(actions))+"\n")      
    for i in range(len(actions)):
        file.write(actions[i]+"\n")
    print ("Number of nodes expanded: ", num_nodes_expanded)

# If algo = Goal Stack Planning        
elif(algo == "g"):            
    actions = GoalStackPlanning()
    print ( "Goal Stack Planning used") 
    print ("Solution Length: ", len(actions))
    file.write(str(len(actions))+"\n")      
    for i in range(len(actions)):
        file.write(actions[i]+"\n")

end = time.time()
print ("Time Taken = ", end - start)

            
file.close()

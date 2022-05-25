WORLD_DIMS = 3, 3

BASE_MODEL = """
% Agent Positions (X,Y,time)
{}

% Possible actions: up, down, right, left
action(up):-agent(X,Y,T), agent(X1,Y1,T1), 
            X1 is X,
            Y1 is Y+1,
            T1 is T+1.
action(down):-agent(X,Y,T), agent(X1,Y1,T1), 
            X1 is X,
            Y1 is Y-1,
            T1 is T+1.
action(right):-agent(X,Y,T), agent(X1,Y1,T1), 
            X1 is X+1,
            Y1 is Y,
            T1 is T+1.
action(left):-agent(X,Y,T), agent(X1,Y1,T1), 
            X1 is X-1,
            Y1 is Y,
            T1 is T+1.
            
            
% The agent_dict must do exactly one step at the time (XOR)
% There is no need for a XOR, since each action is already mutually exclusive by definition -> we need a OR to avoid doing no action
constraint :- action(up);action(down);action(right);action(left).
"""

FACTS_BASE_MODEL = """
{}::agent(0,0,0);
{}::agent(1,0,0);
{}::agent(2,0,0);
{}::agent(2,1,0);
{}::agent(1,1,0);
{}::agent(0,1,0);
{}::agent(0,2,0);
{}::agent(1,2,0);
{}::agent(2,2,0).

{}::agent(0,0,1);
{}::agent(1,0,1);
{}::agent(2,0,1);
{}::agent(2,1,1);
{}::agent(1,1,1);
{}::agent(0,1,1);
{}::agent(0,2,1);
{}::agent(1,2,1);
{}::agent(2,2,1).
"""

BASE_QUERIES = ['action(up)',
                'action(down)',
                'action(right)',
                'action(left)']

BASE_QUERIES_WITH_CONSTRAINT = ['action(up)',
                                'action(down)',
                                'action(right)',
                                'action(left)',
                                'constraint']

TASK_GENERALIZATION_1 = """
% Initial state
{}
  
% The policy is part of the program
action(X,Y,up,up_right) :- Y<2.
action(X,Y,right,up_right) :- Y=2.
action(X,Y,right,right_up) :- X<2.
action(X,Y,up,right_up) :- X=2.


% Transition function
agent(X,Y,0,_) :- agent(X,Y,0).
agent(X,Y,T, Strategy) :- Tprev is T -1, Tprev >=0, Yprev is Y - 1, agent(X,Yprev,Tprev, Strategy), action(X,Yprev,up,Strategy).
agent(X,Y,T, Strategy) :- Tprev is T -1, Tprev >=0, Xprev is X - 1, agent(Xprev,Y,Tprev, Strategy), action(Xprev,Y,right,Strategy).
agent(2,2,T, Strategy) :- Tprev is T -1, Tprev >=0, agent(2,2,Tprev, Strategy).
"""

FACTS_TASK_GEN_1 = """
{}::agent(0,0,0);
{}::agent(1,0,0);
{}::agent(2,0,0);
{}::agent(2,1,0);
{}::agent(1,1,0);
{}::agent(0,1,0);
{}::agent(0,2,0);
{}::agent(1,2,0);
{}::agent(2,2,0).
"""

QUERIES_TASK_GEN_1_UP_RIGHT = """
query(agent(0,0,1, up_right)).
query(agent(0,1,1, up_right)).
query(agent(0,2,1, up_right)).
query(agent(1,0,1, up_right)).
query(agent(1,1,1, up_right)).
query(agent(1,2,1, up_right)).
query(agent(2,0,1, up_right)).
query(agent(2,1,1, up_right)).
query(agent(2,2,1, up_right)).

query(agent(0,0,2, up_right)).
query(agent(0,1,2, up_right)).
query(agent(0,2,2, up_right)).
query(agent(1,0,2, up_right)).
query(agent(1,1,2, up_right)).
query(agent(1,2,2, up_right)).
query(agent(2,0,2, up_right)).
query(agent(2,1,2, up_right)).
query(agent(2,2,2, up_right)).

query(agent(0,0,3, up_right)).
query(agent(0,1,3, up_right)).
query(agent(0,2,3, up_right)).
query(agent(1,0,3, up_right)).
query(agent(1,1,3, up_right)).
query(agent(1,2,3, up_right)).
query(agent(2,0,3, up_right)).
query(agent(2,1,3, up_right)).
query(agent(2,2,3, up_right)).

query(agent(0,0,4, up_right)).
query(agent(0,1,4, up_right)).
query(agent(0,2,4, up_right)).
query(agent(1,0,4, up_right)).
query(agent(1,1,4, up_right)).
query(agent(1,2,4, up_right)).
query(agent(2,0,4, up_right)).
query(agent(2,1,4, up_right)).
query(agent(2,2,4, up_right)).
"""

QUERIES_TASK_GEN_1_RIGHT_UP = """
query(agent(0,0,1, right_up)).
query(agent(0,1,1, right_up)).
query(agent(0,2,1, right_up)).
query(agent(1,0,1, right_up)).
query(agent(1,1,1, right_up)).
query(agent(1,2,1, right_up)).
query(agent(2,0,1, right_up)).
query(agent(2,1,1, right_up)).
query(agent(2,2,1, right_up)).

query(agent(0,0,2, right_up)).
query(agent(0,1,2, right_up)).
query(agent(0,2,2, right_up)).
query(agent(1,0,2, right_up)).
query(agent(1,1,2, right_up)).
query(agent(1,2,2, right_up)).
query(agent(2,0,2, right_up)).
query(agent(2,1,2, right_up)).
query(agent(2,2,2, right_up)).

query(agent(0,0,3, right_up)).
query(agent(0,1,3, right_up)).
query(agent(0,2,3, right_up)).
query(agent(1,0,3, right_up)).
query(agent(1,1,3, right_up)).
query(agent(1,2,3, right_up)).
query(agent(2,0,3, right_up)).
query(agent(2,1,3, right_up)).
query(agent(2,2,3, right_up)).

query(agent(0,0,4, right_up)).
query(agent(0,1,4, right_up)).
query(agent(0,2,4, right_up)).
query(agent(1,0,4, right_up)).
query(agent(1,1,4, right_up)).
query(agent(1,2,4, right_up)).
query(agent(2,0,4, right_up)).
query(agent(2,1,4, right_up)).
query(agent(2,2,4, right_up)).
"""
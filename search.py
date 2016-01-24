# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

"P2-1"
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "[Project 2] YOUR CODE HERE"
    #util.raiseNotDefined()
    from util import Stack
    solution = []          #direction
    to_visit = Stack()     #position
    visited = set()        #already visit
    to_visit.push(problem.getStartState())
    visited.add(problem.getStartState())
    while not to_visit.isEmpty():
        v = to_visit.pop()
        if problem.isGoalState(v):
            return solution
        else:
            stuck = 1
            for x in problem.getSuccessors(v):    
                if x[0] not in visited:
                    stuck = 0
                    solution.append(x[1])
                    to_visit.push(v)
                    to_visit.push(x[0])
                    visited.add(x[0])
                    break
            if stuck:
                solution.pop() 
"P2-2"
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "[Project 2] YOUR CODE HERE"
    #util.raiseNotDefined()
    from util import Queue
    from game import Directions
    solution = []          #direction
    to_visit = Queue()     #position
    visited = set()        #already visit
    cost = {}              #cost of every visited point
    cost[problem.getStartState()] = 0
    to_visit.push(problem.getStartState())
    #visited.add(problem.getStartState())
    startPoint = problem.getStartState()
    while not to_visit.isEmpty():
        v = to_visit.pop()
        visited.add(v)
        if problem.isGoalState(v):
            while v != startPoint:
                minimum = 99999
                temp = problem.getSuccessors(v)[0]
                for x in problem.getSuccessors(v):
                    if x[0] in visited and cost[x[0]] < minimum:
                        minimum = cost[x[0]]
                        temp = x
                v = temp[0]
                
                if temp[1] == Directions.EAST:
                    solution.append(Directions.WEST)
                elif temp[1] == Directions.WEST:
                    solution.append(Directions.EAST)
                elif temp[1] == Directions.NORTH:
                    solution.append(Directions.SOUTH)
                else:
                    solution.append(Directions.NORTH)
            solution.reverse()
            #print solution
            return solution
        else:
            for x in problem.getSuccessors(v):
                if x[0] not in visited:
                    to_visit.push(x[0])
                if x[0] not in cost:
                    cost[x[0]] = cost[v] + x[2]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

"P2-3"
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "[Project 2] YOUR CODE HERE"
    #util.raiseNotDefined()
    from util import PriorityQueue
    from game import Directions
    solution = []
    closeset = set()
    openset = set()
    openq = PriorityQueue()
    came_from = {}
    g = {}
    f = {}
    g[problem.getStartState()] = 0
    f[problem.getStartState()] = g[problem.getStartState()] + heuristic(problem.getStartState(), problem)
    openq.push(problem.getStartState(), f[problem.getStartState()])
    openset.add(problem.getStartState())

    while not openq.isEmpty():
        current = openq.pop()
        if problem.isGoalState(current):
            #solution.append(current)
            while current in came_from:
                if came_from[current][0] - current[0] == 1:
                    solution.append(Directions.WEST)
                elif current[0] - came_from[current][0] == 1:
                    solution.append(Directions.EAST)
                elif current[1] - came_from[current][1] == 1:
                    solution.append(Directions.NORTH)
                else:
                    solution.append(Directions.SOUTH)
                current = came_from[current]
                #solution.append(current)
            solution.reverse()
            #print solution
            return solution
        else:
            closeset.add(current)
            for x in problem.getSuccessors(current):
                if x[0] in closeset:
                    continue
                temp_g = g[current] + x[2]

                if x[0] not in openset or temp_g < g[x[0]]:
                    came_from[x[0]] = current
                    g[x[0]] = temp_g
                    f[x[0]] = g[x[0]] + heuristic(x[0], problem)
                    if x[0] not in openset:
                        openq.push(x[0], f[x[0]])
                        openset.add(x[0])
    return []
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

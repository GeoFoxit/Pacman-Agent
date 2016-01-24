# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

actionList = []

"P3-1"
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	
		"[Project 3] YOUR CODE HERE"
		#if successorGameState.isWin(): return float("inf") - 20
		ghostposition = currentGameState.getGhostPosition(1)
		distfromghost = util.manhattanDistance(ghostposition,newPos)
		score = max(distfromghost,3) + successorGameState.getScore()
		foodlist = newFood.asList()
		closestfood  = 100
		for foodpos in foodlist:
			distnow = util.manhattanDistance(foodpos,newPos)
			if (distnow < closestfood):
				closestfood = distnow
			if (currentGameState.getNumFood() > successorGameState.getNumFood()):
				score+=100
			if action == Directions.STOP:
				score -=3
			score -=3 *closestfood
			capsuleplaces = currentGameState.getCapsules()
			if successorGameState.getPacmanPosition() in capsuleplaces:
				score += 120
			return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

"P3-2"
class MinimaxAgent(MultiAgentSearchAgent):

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.generateSuccessor(agentIndex, action):
		Returns the successor game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game
		"""
		
		"""
		for a in self.miniMax(gameState, self.depth):
			print a
		"""
		#self.miniMax return a tuple contain [score , action]
		return self.miniMax(gameState, self.depth)[1]
		
	def miniMax(self, gameState, depth, agentIndex=0):
		"[Project 3] YOUR CODE HERE"
		
		#: check if the game ends, or we reach the depth
		if gameState.isWin() or gameState.isLose() or depth == 0:
		#: return (current_score, )	
			"""
			print self.evaluationFunction(gameState)
			comma behind self.evaluationFunction(gameState) is for the tuple in actionList 
			tuple contain score,action 
			"""
			return ( self.evaluationFunction(gameState), )
			#: use evaluationFunction calculate score
			
		numAgents = gameState.getNumAgents()
		#: if current agent is the last agent in game, decrease the depth
		newDepth = depth if agentIndex != numAgents - 1 else depth - 1
		
		#: iterate through all depths and agents in the tree
		newAgentIndex = (agentIndex + 1) % numAgents

		#: actionlist (score + action pair list)
		#: recursive call miniMax to generate actionList miniMax produce score given agentIndex , gamestate ,depth and action 
		
		actionList = [ (self.miniMax(gameState.generateSuccessor(agentIndex, a), newDepth, newAgentIndex)[0], a) for a in gameState.getLegalActions(agentIndex)]

		if(agentIndex == 0):    #: max node (pacman)
			#print str(max(actionList)) + str(depth)
			return max(actionList) 
		else:                   #: min node
			return min(actionList)  #: return action that gives min score
		
"P3-3"
class AlphaBetaAgent(MultiAgentSearchAgent):

	def getAction(self, gameState):
		return self.alphaBeta(gameState, self.depth)[1]

	def alphaBeta(self, gameState, depth, agentIndex=0,alpha=-999999,beta=999999):
		#: check if the game ends, or we reach the depth
		if gameState.isWin() or gameState.isLose() or depth == 0:
		#: return (current_score, )	
			"""
			print self.evaluationFunction(gameState)
			comma behind self.evaluationFunction(gameState) is for the tuple in actionList 
			tuple contain score,action 
			"""
			return ( self.evaluationFunction(gameState), )
			#: use evaluationFunction calculate score
		numAgents = gameState.getNumAgents()
		#: if current agent is the last agent in game, decrease the depth
		newDepth = depth if agentIndex != numAgents - 1 else depth - 1
		
		#: iterate through all depths and agents in the tree
		newAgentIndex = (agentIndex + 1) % numAgents

		#: actionlist (score + action pair list)
		#: recursive call miniMax to generate actionList miniMax produce score given agentIndex , gamestate ,depth and action 
		
		actionList = []
		if(agentIndex == 0):    #: max node (pacman)
			#print str(max(actionList)) + str(depth)
			for a in gameState.getLegalActions(agentIndex):
				b = self.alphaBeta(gameState.generateSuccessor(agentIndex, a), newDepth, newAgentIndex, alpha, beta)[0]
				actionList.append([b,a]) 
				alpha = max(b,alpha)
				if (alpha > beta):
					break
			return max(actionList) 
		else:                   #: min node
			for a in gameState.getLegalActions(agentIndex):
				b = self.alphaBeta(gameState.generateSuccessor(agentIndex, a), newDepth, newAgentIndex, alpha, beta)[0]
				actionList.append([b,a]) 
				beta = min(b,beta)
				if (alpha > beta):
					break
			return min(actionList)  #: return action that gives min score
     


"P3-4 Side Mission (optional)"
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()

"P3-4"
def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function.

	  DESCRIPTION: <write something here so we know what you did>
	"""
	
	"[Project 3] YOUR CODE HERE"

	newPos = currentGameState.getPacmanPosition()
	newFood = currentGameState.getFood().asList()
	newGhostStates = currentGameState.getGhostStates()
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

	if currentGameState.isLose(): 
		return -float("inf")
	elif currentGameState.isWin():
		return float("inf")

	# food distance
	foodDist = min(map(lambda x: util.manhattanDistance(newPos, x), newFood))
	
	# number of cap
	numcap = len(currentGameState.getCapsules())

	# number of food left
	numfoodleft = len(newFood)

	# ghost
	ghostScore = 0
	if newScaredTimes[0] > 0:
			  ghostScore += 100.0 # the ghost pacman can eat
	for state in newGhostStates:
		dist = manhattanDistance(newPos, state.getPosition())
		if state.scaredTimer == 0 and dist < 3:
			ghostScore -= 1.0 / (3.0 - dist); # the ghost is going to kill pacman, run away !
		elif state.scaredTimer < dist:
			ghostScore += 1.0 / (dist) # the ghost pacman can eat or it's still far
	
	score =  1 * currentGameState.getScore() - (1.5*foodDist) + ghostScore - (20*numcap) - (4*numfoodleft)
			 # the current score
			 # the far the food is, the willingless i want
			 # the condition of ghost
			 # I really hope that the pacman could eat the cap
			 # cause it could be more possibly for me to pass the game
			 # minimize the food left

	return score
	# Abbreviation
better = betterEvaluationFunction


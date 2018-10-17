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

        "*** YOUR CODE HERE ***"
        # algorithm:
        # first, find the nearest food to the Pacman and the distance to ghost
        # second, modify the evaluation function by adding these two factors
        shortdis = 1000
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    temp = manhattanDistance((x, y), newPos)
                    if temp < shortdis:
                        shortdis = temp
        statescore = successorGameState.getScore()
        newGhostPos = successorGameState.getGhostPositions()
        dis_to_ghost = 0
        for ghostPos in newGhostPos:
            dis_to_ghost += manhattanDistance(newPos, ghostPos)
        retval = statescore + 10/shortdis - 2/(dis_to_ghost+1)
        if action == 'Stop':
            retval -= 10
        return retval
        # return successorGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    """
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    """

    # self helper function
    def getvalue(self, currentGameState, agentIndex, depth, num_agent):
        # base case
        if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
        # if now we are exploring Pacman
        if agentIndex == 0:
            # do something
            legalMoves = currentGameState.getLegalActions(0)
            best_score = -float('inf')
            best_action = ''
            for action in legalMoves:
                next_state = currentGameState.generateSuccessor(0, action)
                score = self.getvalue(next_state, 1, depth, num_agent)
                if score[0] > best_score:
                    best_score = score[0]
                    best_action = action
            return (best_score, best_action)
        # else, we are exploring ghost
        else:
            legalMoves = currentGameState.getLegalActions(agentIndex)
            least_score = float('inf')
            least_action = ''
            for action in legalMoves:
                next_state = currentGameState.generateSuccessor(agentIndex, action)
                if agentIndex == num_agent-1:
                    if depth < self.depth - 1:
                        score = self.getvalue(next_state, 0, depth+1, num_agent)
                    else:
                        score = (self.evaluationFunction(next_state), action)
                else:
                    score = self.getvalue(next_state, agentIndex+1, depth, num_agent)
                if score[0] < least_score:
                    least_score = score[0]
                    least_action = action
            return (least_score, least_action)

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        num_agent = gameState.getNumAgents()
        (score, action) = self.getvalue(gameState, 0, 0, num_agent)
        return action

#       util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # self helper function
    # def getvalue(self, currentGameState, alpha, beta, agentIndex, depth, num_agent):
    #     # base case
    #     if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
    #     # if now we are exploring Pacman
    #     if agentIndex == 0:
    #         return self.max_value(currentGameState, alpha, beta)
    #     # else, we are exploring ghost
    #     else:
            

    def max_value(self, currentGameState, alpha, beta, depth, num_agent):
        if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
        v = -float('inf')
        legalMoves = currentGameState.getLegalActions(0)
        max_action = ''
        for action in legalMoves:
            next_state = currentGameState.generateSuccessor(0, action)
            next_value = self.min_value(next_state, alpha, beta, 1, depth, num_agent)[0]
            if (next_value > v):
                max_action = action
                v = next_value
            if v > beta: return (v, '')
            alpha = max(alpha, v)
        return (v, max_action)


    def min_value(self, currentGameState, alpha, beta, agentIndex, depth, num_agent):
        if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
        v = float('inf')
        legalMoves = currentGameState.getLegalActions(agentIndex)
        min_action = ''
        for action in legalMoves:
            next_state = currentGameState.generateSuccessor(agentIndex, action)
            # base case
            # if this min_node is the last one we need to evaluate, we just return the evaluation function
            if depth == self.depth-1 and agentIndex == num_agent-1:
                next_value = self.evaluationFunction(next_state)
            # if this is not the last agent in the game
            elif agentIndex < num_agent-1:
                next_value = self.min_value(next_state, alpha, beta, agentIndex+1, depth, num_agent)[0]
            # if this is the last agent in the game in one depth
            else:
                next_value = self.max_value(next_state, alpha, beta, depth+1, num_agent)[0]
            if (next_value < v):
                min_action = action
                v = next_value
            if v < alpha: return (v, '')
            beta = min(beta, v)
        return (v, min_action)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = -float('inf')
        beta = float('inf')
        num_agent = gameState.getNumAgents()
        (score, action) = self.max_value(gameState, alpha, beta, 0, num_agent)
        return action
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def max_value(self, currentGameState, depth, num_agent):
        if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
        legalMoves = currentGameState.getLegalActions(0)
        max_action = ''
        max_value = -float('inf')
        for action in legalMoves:
            next_state = currentGameState.generateSuccessor(0, action)
            next_value = self.average_value(next_state, 1, depth, num_agent)[0]
            if (next_value > max_value):
                max_action = action
                max_value = next_value
        return (max_value, max_action)

    def average_value(self, currentGameState, agentIndex, depth, num_agent):
        if currentGameState.isWin() or currentGameState.isLose(): return (currentGameState.getScore(), '')
        legalMoves = currentGameState.getLegalActions(agentIndex)
        # use a value_list to store all possible values and give the average to the state
        value_list = []
        for action in legalMoves:
            next_state = currentGameState.generateSuccessor(agentIndex, action)
            # base case
            # if this min_node is the last one we need to evaluate, we just return the evaluation function
            if depth == self.depth-1 and agentIndex == num_agent-1:
                next_value = self.evaluationFunction(next_state)
            # if this is not the last agent in the game
            elif agentIndex < num_agent-1:
                next_value = self.average_value(next_state, agentIndex+1, depth, num_agent)[0]
            # if this is the last agent in the game in one depth
            else:
                next_value = self.max_value(next_state, depth+1, num_agent)[0]
            value_list.append(next_value)
        average = 1.0*sum(value_list) / len(value_list)
        return (average, '')


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        (score, action) = self.max_value(gameState, 0, num_agent)
        return action
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    """
    In this problem, we consider following factors that affect the evaluation
    1. distance to the nearest food
    2. distance to each ghost
    3. remain scared time for each ghost
    4. current state score
    """

    "*** YOUR CODE HERE ***"
    # 1. distance to the nearest food
    pac2food = float('inf')
    food = currentGameState.getFood()
    pacman_position = currentGameState.getPacmanPosition()
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                temp = manhattanDistance((x, y), pacman_position)
                if temp < pac2food:
                    pac2food = temp
    # 2. distance to each ghost
    # ghost_position is a list contains all ghosts position
    ghosts_position = currentGameState.getGhostPositions()
    dis_to_ghost = [manhattanDistance(pacman_position, ghost_position) for ghost_position in ghosts_position]
    # 3. remain scared time
    ghosts_state = currentGameState.getGhostStates()
    remain_scared_time = []
    for ghost_state in ghosts_state:
        remain_scared_time.append(ghost_state.scaredTimer)
    # 4. current state score
    statescore = currentGameState.getScore()
    # at last, combine all info we have together
    retval = statescore + 10.0/pac2food
    for index in range(1, currentGameState.getNumAgents()):
        # if this ghost is not scared right now, we subtract
        if remain_scared_time[index-1] == 0:
            retval = retval - 2.0/(dis_to_ghost[index-1]+1)
        # else, we reward pacman if the pacman is closer to ghost
        else:
            retval = retval + 6.0/((dis_to_ghost[index-1]+1) / remain_scared_time[index-1])
    return retval
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

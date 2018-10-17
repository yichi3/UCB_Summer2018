# myAgentP3.py
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########
class myAgentP3(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  """

  def __init__( self, index, timeForComputing = .1):
    CaptureAgent.__init__( self, index, timeForComputing = .1)
    self.depth = 3
    self.weights = [1, 1, 0, 0]

  def setTeammatePosition(self, gameState, teammateActions):
    # this helper function take the teammateAction and current gamestate
    # give the future teammate position
    teammatePosition = []
    if teammateActions == None:
      return teammatePosition
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    x, y = gameState.getAgentPosition(teammateIndex)
    for a in teammateActions:
      if a == 'Stop': continue
      if a == 'North': y += 1
      elif a == 'South': y -= 1
      elif a == 'West': x -= 1
      else: x += 1
      teammatePosition.append((x, y))
    return teammatePosition




  def max_My_value(self, currentGameState, depth, myIndex, teammateIndex, teammateActions):
    # max_value get the max value and correspond action made by ourselves
    if currentGameState.isOver(): return (currentGameState.getScore(), [])
    legalMoves = currentGameState.getLegalActions(myIndex)
    filteredActions = actionsWithoutReverse(actionsWithoutStop(legalMoves), currentGameState, self.index)
    max_action_list = []
    max_action = ''
    max_value = -float('inf')
    # two situations
    # 1. if depth is reached, we need to do the evaluation for different actions
    # 2. if depth is not reached, we need to pass the calculation to the next depth
    if depth == self.depth:
      values = [self.evaluate(currentGameState, a) for a in filteredActions]
      max_value = max(values)
      maxIndices = [index for index in range(len(values)) if values[index] == max_value]
      chosenIndex = random.choice(maxIndices)
      max_action = filteredActions[chosenIndex]
      max_action_list.append(max_action)
      return (max_value, max_action_list)
    else:
      for action in filteredActions:
        next_state = currentGameState.generateSuccessor(myIndex, action)
        next_value, next_action_list = self.max_T_value(next_state, depth, myIndex, teammateIndex, teammateActions)
        if (next_value > max_value):
          max_action = action
          max_value = next_value
          max_action_list = next_action_list
      max_action_list.insert(0, max_action)
      return (max_value, max_action_list)

    # filteredActions = actionsWithoutReverse(actionsWithoutStop(legalMoves), currentGameState, self.index)
    # max_action = ''
    # max_value = -float('inf')
    # for action in filteredActions:
    #   next_state = currentGameState.generateSuccessor(myIndex, action)
    #   next_value = self.average_value(next_state, depth, myIndex, teammateIndex, teammateActions)[0]
    #   if (next_value > max_value):
    #     max_action = action
    #     max_value = next_value
    # return (max_value, max_action)

  def max_T_value(self, currentGameState, depth, myIndex, teammateIndex, teammateActions):
    if currentGameState.isOver(): return (currentGameState.getScore(), [])
    legalMoves = currentGameState.getLegalActions(teammateIndex)
    value_list = []
    for action in legalMoves:
      if len(teammateActions) > depth and teammateActions[depth] in legalMoves:
        action = teammateActions[depth]
      next_state = currentGameState.generateSuccessor(teammateIndex, action)
      next_value = self.max_My_value(next_state, depth+1, myIndex, teammateIndex, teammateActions)
      value_list.append(next_value)
      if len(teammateActions) > depth and teammateActions[depth] in legalMoves:
        break
    # now we need to choose the max value and its action
    max_value = -float('inf')
    max_action_list = []
    for value in value_list:
      if max_value < value[0]:
        max_value = value[0]
        max_action_list = value[1]
    # return the max value and action
    return (max_value, max_action_list)


  # def average_value(self, currentGameState, depth, myIndex, teammateIndex, teammateActions):
  #   # in this function, we will detect the game tree by applying teammate action
  #   # if no teammate action is detected, we will do the expected max algorithm
  #   if currentGameState.isOver(): return (currentGameState.getScore(), '')

  #   legalMoves = currentGameState.getLegalActions(teammateIndex)
  #   # use a value_list to store all possible values and give the average to the state
  #   value_list = []
  #   for action in legalMoves:
  #     if not teammateActions == None:
  #       if len(teammateActions) > depth and teammateActions[depth] in legalMoves:
  #         action = teammateActions[depth]
  #     next_state = currentGameState.generateSuccessor(teammateIndex, action)
  #     # base case
  #     # if this min_node is the last one we need to evaluate, we just return the evaluation function
  #     if depth == self.depth-1:
  #       next_value = (self.evaluate(next_state, teammateIndex), '')
  #     # if this is the last agent in the game in one depth
  #     else:
  #       next_value = self.max_value(next_state, depth+1, myIndex, teammateIndex, teammateActions)
  #     value_list.append(next_value)
  #     # if len(teammateActions) > depth:
  #     #   break
  #   SUM = 0
  #   for value in value_list:
  #     SUM +=  value[0]
  #   average = SUM / len(value_list)
  #   return (average, '')

    # # if the depth is reached the required self.depth
    # if depth == self.depth:
    #   # if we can get the action taken by teammate
    #   if len(teammateActions) > depth:
    #     teammateAction = teammateActions[depth]
    #     next_value = self.evaluate(currentGameState, teammateAction, teammateIndex)
    #     return (next_value, '')
    #   else:
    #     legalMoves = currentGameState.getLegalActions(agentIndex)
    # # first situation, if we can derive the action given by teammate
    # # the only thing we need to do is to process the action taken by teammate and go to max_node
    # if len(teammateActions) > depth:
    #   next_state = currentGameState.generateSuccessor(teammateIndex, teammateActions[depth])
    #   next_value = self.max_value(next_state, depth+1, num_agent, myIndex, teammateIndex, teammateActions)[0]
    # elif:


    


  def getMyAction(self, gameState, teammateActions, depth):
    """
      Returns the expectimax action using and self.evaluate()
    """
    # in this helper function, we will broadcast our future plan to teammate
    # we limit our search in a depth of 5
    # this algorithm can be considered as a game tree. We traverse the tree and evaluate the value at leaves
    # choose the largest value and return the path
    # get teammate index
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    # get our index
    myIndex = self.index
    (score, action) = self.max_My_value(gameState, 0, myIndex, teammateIndex, teammateActions)
    return action


  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line. 
    # If you would like to use Manhattan distances instead 
    # of maze distances in order to save on initialization 
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    teammateActions = self.receivedBroadcast
    # Process your teammate's broadcast! 
    # Use it to pick a better action for yourself

    actions = gameState.getLegalActions(self.index)

    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

    # this function gives the future plan of teammate, which helps us to do a better choice of our action
    teammatePosition = self.setTeammatePosition(gameState, teammateActions)
    # evaulate values for each action

    if teammateActions == None:
      teammateActions = []
    chosenAction = self.getMyAction(gameState, teammateActions, 0)

    # this part broadcast our future steps for our teammate
    futureActions = chosenAction[1:len(chosenAction)]
    self.toBroadcast = futureActions
    return chosenAction[0]

    # values = [self.evaluate(gameState, a, teammatePosition) for a in actions]
    # # penalize Stop action
    # for a in actions:
    #   if a == 'Stop':
    #     values[actions.index(a)] -= 50
    # # choose the max value among all values
    # max_value = max(values)
    # maxIndices = [index for index in range(len(values)) if values[index] == max_value]
    # chosenIndex = random.choice(maxIndices)

    # # this part broadcast our future steps for our teammate
    # futureActions = None
    # self.toBroadcast = futureActions
    # return actions[chosenIndex]


  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    # successorGameState = gameState.generateSuccessor(self.index, action)
    # return self.evaluationFunction(successorGameState)
    features = self.getFeatures(gameState, action)
    weights = self.getWeights()
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()

    ### Useful information you can extract from a GameState (pacman.py) ###
    successorGameState = gameState.generateSuccessor(self.index, action)
    newPos = successorGameState.getAgentPosition(self.index)
    oldFood = gameState.getFood()
    newFood = successorGameState.getFood()
    ghostIndices = self.getOpponents(successorGameState)

    # Determines how many times the agent has already been in the newPosition in the last 20 moves
    numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

    foodPositions = newFood.asList()
    foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
    closestFood = min( foodDistances ) + 1.0

    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths

    # CHANGE YOUR FEATURES HERE
    features['nearest_food'] = 1./closestFood
    features['number_food'] = len(foodPositions)

    features['teammate_distance'] = teammateDistance

    features['ghost_distance'] = closestGhost
    features['num_repeat'] = 1./(numRepeats+1)

    return features

  def getWeights(self):
    # CHANGE YOUR WEIGHTS HERE
    weights = util.Counter()
    # weights['successorScore'] = 200
    weights['nearest_food'] = 1.0
    weights['number_food'] = -1.0
    # weights['total_distance'] = 1.0
    # weights['teammate_distance'] = 0.2
    # weights['ghost_distance'] = 0.3
    # weights['num_repeat'] = 10
    return weights

  # def evaluationFunction(self, state):
  #   foods = state.getFood().asList()
  #   ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
  #   friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]

  #   pacman = state.getAgentPosition(self.index)

  #   closestFood = min(self.distancer.getDistance(pacman, food) for food in foods) + 2.0 \
  #       if len(foods) > 0 else 1.0
  #   closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) + 1.0 \
  #       if len(ghosts) > 0 else 1.0
  #   closestFriend = min(self.distancer.getDistance(pacman, friend) for friend in friends) + 1.0 \
  #       if len(friends) > 0 else 1.0

  #   closestFoodReward = 1.0 / closestFood
  #   closestGhostPenalty = 1.0 / (closestGhost ** 2) if closestGhost < 20 else 0
  #   closestFriendPenalty = 1.0 / (closestFriend ** 2) if closestFriend < 5 else 0

  #   numFood = len(foods)

  #   features = [-numFood, closestFoodReward, closestGhostPenalty, closestFriendPenalty]

  #   value = sum(feature * weight for feature, weight in zip(features, self.weights))
  #   return value


  # def evaluate(self, gameState, teammateIndex):
  #   """
  #   Computes a linear combination of features and feature weights
  #   """
  #   features = self.getFeatures(gameState, teammateIndex)
  #   weights = self.getWeights(gameState)
  #   return features * weights

  # def getFeatures(self, gameState, teammateIndex):
  #   features = util.Counter()

  #   ### Useful information you can extract from a GameState (pacman.py) ###
  #   newPos = gameState.getAgentPosition(self.index)
  #   oldFood = gameState.getFood()
  #   ghostIndices = self.getOpponents(gameState)

  #   # Determines how many times the agent has already been in the newPosition in the last 20 moves
  #   numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

  #   foodPositions = oldFood.asList()
  #   foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
  #   closestFood = min( foodDistances ) + 1.0

  #   ghostPositions = [gameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
  #   ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
  #   ghostDistances.append( 1000 )
  #   closestGhost = min( ghostDistances ) + 1.0

  #   teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
  #   assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
  #   teammateIndex = teammateIndices[0]
  #   teammatePos = gameState.getAgentPosition(teammateIndex)
  #   teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

  #   pacmanDeath = gameState.data.num_deaths

  #   features['Score'] = self.getScore(gameState)

  #   # CHANGE YOUR FEATURES HERE
  #   features['nearest_food'] = 1./closestFood
  #   total_distance = sum(foodDistances)
  #   features['total_distance'] = 1./total_distance
  #   features['teammate_distance'] = teammateDistance

  #   features['ghost_distance'] = closestGhost
  #   features['num_repeat'] = 1./(numRepeats+1)

  #   return features

  # def getWeights(self, gameState):
  #   # CHANGE YOUR WEIGHTS HERE
  #   weights = util.Counter()
  #   weights['Score'] = 200
  #   weights['nearest_food'] = 100
  #   weights['total_distance'] = 20
  #   weights['teammate_distance'] = 0.2
  #   weights['ghost_distance'] = 0.3
  #   weights['num_repeat'] = 10
  #   return weights


  # def evaluate(self, gameState, action, teammatePosition):
  #   """
  #   Computes a linear combination of features and feature weights
  #   """
  #   features = self.getFeatures(gameState, action, teammatePosition)
  #   weights = self.getWeights(gameState, action)
  #   return features * weights

  # def getFeatures(self, gameState, action, teammatePosition):
  #   features = util.Counter()

  #   ### Useful information you can extract from a GameState (pacman.py) ###
  #   successorGameState = gameState.generateSuccessor(self.index, action)
  #   newPos = successorGameState.getAgentPosition(self.index)
  #   oldFood = gameState.getFood()
  #   ghostIndices = self.getOpponents(successorGameState)

  #   # Determines how many times the agent has already been in the newPosition in the last 20 moves
  #   numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

  #   foodPositions = oldFood.asList()
  #   # in this case, we try to remove the food that our teammate will visit and check the remain food
  #   for position in teammatePosition:
  #     if position in foodPositions:
  #       foodPositions.remove(position)

  #   foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
  #   if foodDistances == []:
  #     closestFood = 1.0
  #   else:
  #     closestFood = min( foodDistances ) + 1.0

  #   ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
  #   ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
  #   ghostDistances.append( 1000 )
  #   closestGhost = min( ghostDistances ) + 1.0

  #   teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
  #   assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
  #   teammateIndex = teammateIndices[0]
  #   teammatePos = successorGameState.getAgentPosition(teammateIndex)
  #   teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

  #   pacmanDeath = successorGameState.data.num_deaths

  #   features['successorScore'] = self.getScore(successorGameState)

  #   # CHANGE YOUR FEATURES HERE
  #   features['nearest_food'] = 1./closestFood
  #   total_distance = sum(foodDistances)
  #   features['total_distance'] = 1./total_distance
  #   features['teammate_distance'] = teammateDistance

  #   features['ghost_distance'] = closestGhost
  #   features['num_repeat'] = 1./(numRepeats+1)

  #   return features

  # def getWeights(self, gameState, action):
  #   # CHANGE YOUR WEIGHTS HERE
  #   weights = util.Counter()
  #   weights['successorScore'] = 200
  #   weights['nearest_food'] = 100
  #   weights['total_distance'] = 20
  #   weights['teammate_distance'] = 0.2
  #   # weights['ghost_distance'] = 0.3
  #   weights['num_repeat'] = 10
  #   return weights



def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions
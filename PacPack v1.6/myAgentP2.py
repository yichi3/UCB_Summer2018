# myAgentP2.py
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

# global variable
# teammate_all_position = util.Counter()


class myAgentP2(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  Students' Names: Yichi Zhang & Handi Xie 
  Phase Number: 2
  Description of Bot:
  For this problem, we largerly depend on what we have done in phase 1
  In phase 2, since we have already known where our teammate would be in future, we would let our agent
  go to the place where our teammate wou't go. Then we found that this is not enough to solve the problem
  therefore, we also let our robot go to the food where our teammate would reach later in his schedule
  """

  def __init__( self, index, timeForComputing = .1):
    CaptureAgent.__init__( self, index, timeForComputing = .1)
    self.teammate_all_position = util.Counter()
  

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

    otherAgentActions = self.receivedInitialBroadcast
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1
    teammateIndex = teammateIndices[0]
    otherAgentPositions = getFuturePositions(gameState, otherAgentActions, teammateIndex)
    
    # You can process the broadcast here!

    # we first record the position that our teammate would go
    x, y = gameState.getAgentPosition(teammateIndex)
    order = 1
    self.teammate_all_position = util.Counter()
    self.teammate_all_position[(x, y)] = order
    for a in otherAgentActions:
      if a == 'Stop': continue
      if a == 'North': y += 1
      elif a == 'South': y -= 1
      elif a == 'West': x -= 1
      else: x += 1
      if self.teammate_all_position[(x, y)] == 0:
        self.teammate_all_position[(x, y)] = order+1
        order += 1


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # penalize Stop action
    for a in actions:
      if a == 'Stop':
        values[actions.index(a)] -= 50
    max_value = max(values)
    maxIndices = [index for index in range(len(values)) if values[index] == max_value]
    chosenIndex = random.choice(maxIndices)
    return actions[chosenIndex]

  
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()

    ### Useful information you can extract from a GameState (pacman.py) ###
    successorGameState = gameState.generateSuccessor(self.index, action)
    newPos = successorGameState.getAgentPosition(self.index)
    ghostIndices = self.getOpponents(successorGameState)

    
    # we compare the possible positions for teammate and food to find if our teammate will get a food in future
    food = gameState.getFood()
    food_position = food.asList()
    teammate_food_order = util.Counter()
    remain_food_list = list()
    for position in food_position:
      if position in self.teammate_all_position:
        teammate_food_order[position] = self.teammate_all_position[position]
      else:
        remain_food_list.append(position)

    # we want to go to food which teammate won't go and teammate will go but very late in future
    teammate_food_order_list = teammate_food_order.sortedKeys()
    teammate_number_food = len(teammate_food_order_list)
    # foodPositions is the list of food positions that our pacman need to visit
    foodPositions = []
    for food in remain_food_list:
      foodPositions.append(food)
    for i in range (0, int(teammate_number_food/2)):
      foodPositions.append(teammate_food_order_list[i])
    

    foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
    # print 'food_position'
    # print food_position
    # print 'remain_food_list'
    # print remain_food_list
    # print 'foodPositions'
    # print foodPositions
    # print 'foodDistances'
    # print foodDistances
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

    features['successorScore'] = self.getScore(successorGameState)

    # CHANGE YOUR FEATURES HERE
    features['nearest_food'] = 1./closestFood
    total_distance = sum(foodDistances) + 1
    features['total_distance'] = 1./total_distance
    features['teammate_distance'] = teammateDistance

    return features

  def getWeights(self, gameState, action):
    # CHANGE YOUR WEIGHTS HERE
    weights = util.Counter()
    weights['successorScore'] = 200
    weights['nearest_food'] = 100
    weights['total_distance'] = 20
    weights['teammate_distance'] = 0.02
    return weights


def getFuturePositions(gameState, plannedActions, agentIndex):
  """
  Returns list of future positions given by a list of actions for a
  specific agent starting form gameState

  NOTE: this does not take into account other agent's movements
  (such as ghosts) that might impact the *actual* positions visited
  by such agent
  """
  if plannedActions is None:
    return None

  planPositions = [gameState.getAgentPosition(agentIndex)]
  for action in plannedActions:
    if action in gameState.getLegalActions(agentIndex):
      gameState = gameState.generateSuccessor(agentIndex, action)
      planPositions.append(gameState.getAgentPosition(agentIndex))
    else:
      print("Action list contained illegal actions")
      break
  return planPositions
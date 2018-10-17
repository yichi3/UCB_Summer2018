# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allstate = self.mdp.getStates()
        # we need to do iterations to update the value for each state
        for turns in range(self.iterations):
            temp_counter = util.Counter()
            for eachstate in allstate:
                # 1. check whether this state is the terminal state
                if self.mdp.isTerminal(eachstate):
                    continue
                next_actions = self.mdp.getPossibleActions(eachstate)
                # use a variable the store the max value when going through each action
                max_value = -float('inf')
                for action in next_actions:
                    nextstate_pro = self.mdp.getTransitionStatesAndProbs(eachstate, action)
                    temp = 0.
                    for state_pro_pair in nextstate_pro:
                        nextstate = state_pro_pair[0]
                        pro = state_pro_pair[1]
                        temp += pro * (self.mdp.getReward(eachstate, action, nextstate) + self.discount*self.values[nextstate])
                    max_value = temp if temp > max_value else max_value
                temp_counter[eachstate] = max_value
            # update V values for all state
            for eachstate in allstate:
                self.values[eachstate] = temp_counter[eachstate]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q value is calculated based on k-th value iteration
        # therefore, this Q value is actually Q_{k+1}
        if self.mdp.isTerminal(state):
            return 0
        nextstate_pro = self.mdp.getTransitionStatesAndProbs(state, action)
        retval = 0.
        for state_pro_pair in nextstate_pro:
            nextstate = state_pro_pair[0]
            pro = state_pro_pair[1]
            retval += pro * (self.mdp.getReward(state, action, nextstate) + self.discount*self.values[nextstate])
        return retval
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # special case: TERMINAL_STATE
        if self.mdp.isTerminal(state):
            return None
        # find the action with the largest Q value
        next_actions = self.mdp.getPossibleActions(state)
        Q_value = [self.computeQValueFromValues(state, action) for action in next_actions]
        best_Q_value = max(Q_value)
        # return the first action found by the search
        for index in range(len(Q_value)):
            if Q_value[index] == best_Q_value:
                return next_actions[index]
        return None
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # in this function, we only update value for one state
        allstate = self.mdp.getStates()
        index = 0
        state_number = len(allstate)
        for turns in range(self.iterations):
            # first, find the state that we need to update
            current_state = allstate[index%state_number]
            # update index for next iteration
            index += 1
            # second, update the value for this state, method is the same in ValueIterationAgent
            if self.mdp.isTerminal(current_state):
                continue
            next_actions = self.mdp.getPossibleActions(current_state)
            # use a variable the store the max value when going through each action
            max_value = -float('inf')
            for action in next_actions:
                nextstate_pro = self.mdp.getTransitionStatesAndProbs(current_state, action)
                temp = 0.
                for state_pro_pair in nextstate_pro:
                    nextstate = state_pro_pair[0]
                    pro = state_pro_pair[1]
                    temp += pro * (self.mdp.getReward(current_state, action, nextstate) + self.discount*self.values[nextstate])
                max_value = temp if temp > max_value else max_value
            self.values[current_state] = max_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allstate = self.mdp.getStates()
        state_number = len(allstate)
        # first, we need to find predecessors for each state
        predecessors = [set() for i in range(state_number)]
        for current_state in allstate:
            # find predecessor for each state
            if self.mdp.isTerminal(current_state):
                continue
            next_actions = self.mdp.getPossibleActions(current_state)
            for action in next_actions:
                nextstate_pro = self.mdp.getTransitionStatesAndProbs(current_state, action)
                for state_pro_pair in nextstate_pro:
                    nextstate, pro = state_pro_pair
                    if pro > 0.:
                        n_index = allstate.index(nextstate)
                        predecessors[n_index].add(current_state)

        priorityqueue = util.PriorityQueue()
        # second, push all non-terminal state onto PQ
        for index in range(state_number):
            current_state = allstate[index]
            if self.mdp.isTerminal(current_state):
                continue
            diff = abs(self.values[current_state] - self.maxValue(current_state))
            priorityqueue.update(current_state, -diff)
        # third, begin the iteration
        for turns in range(self.iterations):
            # if PQ is empty, return directly
            if priorityqueue.isEmpty():
                return
            # pop a state s from PQ
            s = priorityqueue.pop()
            # update s value
            self.values[s] = self.maxValue(s)
            # do operations on predecesor p of s
            s_index = allstate.index(s)
            for p in predecessors[s_index]:
                diff = abs(self.values[p] - self.maxValue(p))
                if diff > self.theta:
                    priorityqueue.update(p, -diff)


    # helper function to find the max Q value from one state
    # state cannot be TERMINAL_STATE
    def maxValue(self, current_state):
        next_actions = self.mdp.getPossibleActions(current_state)
        # use a variable the store the max value when going through each action
        max_value = -float('inf')
        for action in next_actions:
            nextstate_pro = self.mdp.getTransitionStatesAndProbs(current_state, action)
            temp = 0.
            for state_pro_pair in nextstate_pro:
                nextstate = state_pro_pair[0]
                pro = state_pro_pair[1]
                temp += pro * (self.mdp.getReward(current_state, action, nextstate) + self.discount*self.values[nextstate])
            max_value = temp if temp > max_value else max_value
        return max_value



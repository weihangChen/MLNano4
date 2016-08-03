import random
from agent import *

class QLearn(object):
    def __init__(self, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def backwardPropagationQValueFromCurrentStateToPreviousState(self, transitionSAQ, currentStateReward, currentState):
        maxqCurrentState = max([x.QValue for x in currentState.SAQs])
        if transitionSAQ.QValue == 0:
            transitionSAQ.QValue = currentStateReward
        else:
            transitionSAQ.QValue = transitionSAQ.QValue + self.alpha * (currentStateReward + self.gamma * maxqCurrentState - transitionSAQ.QValue)

    def getMaxQValue(self, state):
        return max(x.QValue for x in state.SAQs)
    
    def getActionsInQValue(self, state, maxQValue):
        return [x.Action for x in state.SAQs if x.QValue == maxQValue]

    def getActionForMaxQValue(self, state):
        maxq = self.getMaxQValue(state)
        actions = self.getActionsInQValue(state, maxq)
        return random.choice(actions)

    def GetStateCreateIfNotExist(self, agent, inputState, actions):
        foundState = None
        foundStates = [x for x in agent.States if x == inputState]
        if (len(foundStates) == 0):
            foundState = inputState
            for action in actions:
                model = StateActionQValueModel(foundState, action)
                foundState.SAQs.append(model)
            agent.States.append(foundState)
        else:
            foundState = foundStates[0]
        return foundState


   


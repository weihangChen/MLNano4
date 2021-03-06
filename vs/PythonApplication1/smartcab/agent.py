import random
import uuid
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from enum import Enum
import copy
#from algorithm import *


# region resources
# https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
# http://artint.info/html/ArtInt_265.html
# https://www.youtube.com/watch?v=YfSBYf9h7qk&index=4&list=PLKG3ExuC02lutWKromayVdguZGfdaz6K_
# https://discussions.udacity.com/t/stuck-on-q-learning/159161
# https://discussions.udacity.com/t/next-state-action-pair/44902/10
# https://discussions.udacity.com/t/correct-q-value-calculation/174729/2
# https://discussions.udacity.com/t/please-someone-clear-up-a-couple-of-points-to-me/45365/2
# https://github.com/studywolf/blog/blob/master/RL/Cat%20vs%20Mouse%20exploration/qlearn.py
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm
# http://www.wearepop.com/articles/secret-formula-for-self-learning-computers
# endregion

# region model class
class QTableSnabShotPerTrial:
    def __init__(self, states, trialIndex, mistakes):
        self.states = states
        self.trialIndex = trialIndex
        self.mistakes = mistakes
        self.totalNormalizedQ = 0


        
class Result:
    def __init__(self):
        # used to calculate total positive and negative rewards
        self.Rewards = []
        # trial index
        self.TrialCount = 0
        # history of qtables within trials
        self.QSnabShots = []

    def getStats(self):
        positiveSum = sum([x for x in self.Rewards if x > 0])
        negativeSum = sum([x for x in self.Rewards if x < 0])
        msg = 'total positive reward: {0} / total negative reward: {1}'.format(positiveSum, negativeSum)
        return msg

    def printMistakeStats(self):
        mistakeSeries = [x.mistakes for x in self.QSnabShots if len(x.mistakes) > 0]
        mistakeStats = [[x.trialIndex, len(x.mistakes)] for x in self.QSnabShots]
        print mistakeStats

    #see if it converges, how much qtable gets updated
    def NormalizeAndSumQTable(self):
        for snabshot in self.QSnabShots: 
            maxqAllStates = max([x.getMaxQ() for x in snabshot.states])
            #set normalized qvalue
            for state in snabshot.states:
                maxq = state.getMaxQ()
                state.normalizedQ = maxq / maxqAllStates
            #sum up the normalized qvalue
            snabshot.totalNormalizedQ = sum([x.normalizedQ for x in snabshot.states])
            


    # i can only assume the same states value are populated
    def GetNormalizedQValueChangePercentageSeries(self):
        changeSeriesInPercentage = []
        count = len(self.QSnabShots)
        for x in range(0, count - 1):
            snabshot1 = self.QSnabShots[x]
            snabshot2 = self.QSnabShots[x + 1]
         
            changePercentage = int((snabshot2.totalNormalizedQ - snabshot1.totalNormalizedQ) / snabshot1.totalNormalizedQ * 100)
            changeSeriesInPercentage.append('{0}%'.format(changePercentage))
        return changeSeriesInPercentage

        
            

class Location:
    def __init__(self, x, y):
        self.LocationId = uuid.uuid4()
        self.X = x
        self.Y = y
    
    def __eq__(self, other):
        return self.X == other.X and self.Y == other.Y

    def __ne__(self, other):
        return not self.__eq__(other)

class State:
     def __init__(self, next_waypoint, light, left, oncoming,  reward=0):
        self.StateId = uuid.uuid4()
        self.NextWaypoint = next_waypoint
        self.Light = light
        self.Left = left
        self.Oncoming = oncoming
        self.Reward = reward
        self.SAQs = []
        self.maxQ = 0
        self.normalizedQ = 0
        self.IsNewlyCreated = False

     def getMaxQ(self):
        maxq = 0
        if (len(self.SAQs) > 0):
            maxq = max([x.QValue for x in self.SAQs])
        return maxq
     
     def __eq__(self, other):
        isEqual = self.NextWaypoint == other.NextWaypoint and self.Light == other.Light and self.Left == other.Left and self.Oncoming == other.Oncoming
        return isEqual

     def __ne__(self, other):
        return not self.__eq__(other)


     def __str__(self):
        msg = 'waypoint: {0} / light: {1} / oncoming: {2} / left: {3}'.format(self.NextWaypoint, self.Light, self.Oncoming, self.Left)
        return msg

class StateActionQValueModel:
    def __init__(self, state, action, qvalue=0):
        self.StateActionPairId = uuid.uuid4()
        self.State = state
        self.Action = action
        self.QValue = qvalue

    def __eq__(self, other):
        return self.State == other.State and self.Action == other.Action

    def __ne__(self, other):
        return not self.__eq__(other)
#endregion



#region qlearning
class QLearn(object):
    def __init__(self, epsilon=0.1, alpha=0.4, gamma=1):
        #self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.mistakes = []

    def backwardPropagationQValueFromCurrentStateToPreviousState(self, transitionSAQ, currentStateReward, currentState):
        if transitionSAQ.QValue == 0:
            transitionSAQ.QValue = currentStateReward
        else:
            transitionSAQ.QValue = transitionSAQ.QValue + self.alpha * (currentStateReward + self.gamma * currentState.getMaxQ() - transitionSAQ.QValue)

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
            foundState.IsNewlyCreated = True
            for action in actions:
                model = StateActionQValueModel(foundState, action)
                foundState.SAQs.append(model)
            agent.States.append(foundState)
        else:
            foundState = foundStates[0]
            foundState.IsNewlyCreated = False
        return foundState
#endregion
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default
                                                  # color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.QLearn = QLearn(epsilon=0.1, alpha=0.3, gamma=1)
        self.States = []  
        self.Result = Result()
        self.StartTakingSnabShotIndex = 60
        self.SnabShotCalIndex = 99
        self.StartRecordingRewardIndex = 60


    

    def reset(self, destination=None):
        self.planner.route_to(destination) 
        self.mistakes = []
        # TODO: Prepare for a new trip; reset any variables here, if required
        if (self.Result.TrialCount > self.StartTakingSnabShotIndex):             
                self.Result.QSnabShots.append(QTableSnabShotPerTrial(copy.deepcopy(self.States), self.Result.TrialCount, self.mistakes))
        self.Result.TrialCount += 1
        if (self.Result.TrialCount > self.SnabShotCalIndex):
            print self.Result.getStats()
            self.Result.NormalizeAndSumQTable()
            print self.Result.GetNormalizedQValueChangePercentageSeries()
            self.Result.printMistakeStats()

    def getAgentLocation(self):
        return self.env.agent_states[self]['location']

   


    def getStateFromInputs(self, nextwaypoin, light, left, oncoming):
        dummyState = State(nextwaypoin, light, left, oncoming)
        foundState = self.QLearn.GetStateCreateIfNotExist(self, dummyState, self.env.valid_actions)
        return foundState

    def update(self, t):
        try:
            # Gather inputs
            self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
            inputs = self.env.sense(self)
            deadline = self.env.get_deadline(self)
            currentState = self.getStateFromInputs(self.next_waypoint, 
                                                   inputs['light'], 
                                                   inputs['oncoming'],
                                                   inputs['left'])
            
        
            # TODO: Update state
            self.state = currentState

            # TODO: Select action according to your policy
            action = self.QLearn.getActionForMaxQValue(currentState)
            
            # action = random.choice(Environment.valid_actions[1:])
            transitionSAQ = next(x for x in currentState.SAQs if x.Action == action)
            
            # Execute action and get reward
            reward = self.env.act(self, action)

            
            if (self.Result.TrialCount > self.StartRecordingRewardIndex):
                self.Result.Rewards.append(reward)
                
                if (action != None and self.next_waypoint != action):
                    self.mistakes.append([self.Result.TrialCount, t, inputs, self.next_waypoint, action, currentState.IsNewlyCreated])

    
            
            # sense the env again
            self.next_waypoint = self.planner.next_waypoint()
            inputs = self.env.sense(self)
            stateAfterMove = self.getStateFromInputs(self.next_waypoint, 
                                                   inputs['light'], 
                                                   inputs['oncoming'],
                                                   inputs['left'])
        
            # TODO: Learn policy based on state, action, reward
            self.QLearn.backwardPropagationQValueFromCurrentStateToPreviousState(transitionSAQ, reward, stateAfterMove)
        except Exception,e: 
            print str(e)
            raise

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    #e.set_primary_agent(a, enforce_deadline=False) # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)
    # NOTE: You can set enforce_deadline=False while debugging to allow longer
                                                     # trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set
                                                          # display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on
                           # the command-line
if __name__ == '__main__':
    run()



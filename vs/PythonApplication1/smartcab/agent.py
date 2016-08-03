import random
import uuid
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from enum import Enum
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
# endregion

# region model class
class TimeCritical(Enum):
    Positive = 1
    Negative = -1

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
     def __init__(self, location, timecritical=TimeCritical.Negative, reward=0):
        self.StateId = uuid.uuid4()
        self.Location = location
        self.TimeCritical = timecritical
        self.Reward = reward
        self.SAQs = []
     
     def __eq__(self, other):
        return self.Location == other.Location

     def __ne__(self, other):
        return not self.__eq__(other)

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
#endregion




class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default
                                                  # color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
                                                          

        self.QLearn = QLearn(epsilon=0.1, alpha=0.1, gamma=0.5)
        self.CurrentState = None
        self.PreviousState = None
        self.lastsaq = None
        self.States = []
        

    

    def reset(self, destination=None):
        self.planner.route_to(destination) 
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    def getAgentLocation(self):
        return self.env.agent_states[self]['location']

    def getStateFromPosition(self, xy):
        dummyState = State(Location(xy[0], xy[1]))
        foundState = self.QLearn.GetStateCreateIfNotExist(self, dummyState, self.env.valid_actions)
        return foundState

    def update(self, t):
        try:
            # Gather inputs
            self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
            inputs = self.env.sense(self)
            deadline = self.env.get_deadline(self)
            currentState = self.getStateFromPosition(self.getAgentLocation())
            
        
            # TODO: Update state
            # do not know what to update, aren't total amount of states a fixed
            # number = count of traffic light combinations * is time running
            # out ?

            # TODO: Select action according to your policy
            action = self.QLearn.getActionForMaxQValue(currentState)
            transitionSAQ = next(x for x in currentState.SAQs if x.Action == action)
            #action = random.choice(Environment.valid_actions[1:])
            # Execute action and get reward
            reward = self.env.act(self, action)
            stateAfterMove = self.getStateFromPosition(self.getAgentLocation())
        
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
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
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



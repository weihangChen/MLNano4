import unittest
import pytest
#from smartcab.algorithm import QLearn
from smartcab.agent import *

class Test_MDPTest(unittest.TestCase):
    
    def setUp(self):
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.states = []
        self.e = Environment()
        self.agent = self.e.create_agent(LearningAgent)
        self.qlearn = QLearn(epsilon=0.1, alpha=0.1, gamma=1)
        
   
    def test_Equal(self):
        #location, not used in state, will blow up the state space
        l1 = Location(1,2)
        l2 = Location(1,2)
        l3 = Location(1,3)
        self.assertTrue(l1 == l2)
        self.assertTrue(l1 != l3)
        
        #state
        state1 = State(next_waypoint = 'right', light = 'green', left = None, oncoming = None )
        state2 = State(next_waypoint = 'right', light = 'green', left = None, oncoming = None )
        state3 = State(next_waypoint = 'right', light = 'red', left = None, oncoming = None)
        self.assertTrue(state1 == state2)
        self.assertTrue(state2 != state3)
        
        #stateactionqvalue
        model1 = StateActionQValueModel(state1, self.valid_actions[1])
        model2 = StateActionQValueModel(state1, self.valid_actions[1])
        model3 = StateActionQValueModel(state2, self.valid_actions[1])
        model4 = StateActionQValueModel(state1, self.valid_actions[2])
        model5 = StateActionQValueModel(state3, self.valid_actions[1])
        self.assertTrue(model1 == model2)
        self.assertTrue(model1 == model3)
        self.assertTrue(model1 != model4)
        self.assertTrue(model1 != model5)

    def test_QvalueBackwardPropagationCal(self):
        landedState = self.agent.getStateFromInputs(nextwaypoin = 'right', light = 'green', left = None, oncoming = None)
        landedState.SAQs[0].QValue = 5
        landedState.SAQs[1].QValue = 4
        landedState.SAQs[2].QValue = 3
        landedState.SAQs[3].QValue = 5
        landedState.Reward = 10

        previousState = self.agent.getStateFromInputs(nextwaypoin = 'right', light = 'red', left = None, oncoming = None)
        saqprevious = landedState.SAQs[0]
        saqprevious.QValue = 0
        
        #case1
        self.qlearn.backwardPropagationQValueFromCurrentStateToPreviousState(saqprevious, landedState.Reward, landedState)
        self.assertTrue(saqprevious.QValue == 10)
        #case2
        saqprevious.QValue = 3
        self.qlearn.backwardPropagationQValueFromCurrentStateToPreviousState(saqprevious, landedState.Reward, landedState)
        self.assertTrue(saqprevious.QValue == 4.2)


    
    def test_ActionChoose(self):
        actionState = self.agent.getStateFromInputs(nextwaypoin = 'right', light = 'green', left = None, oncoming = None)
        actionState.SAQs[0].QValue = 5
        actionState.SAQs[1].QValue = 4
        actionState.SAQs[2].QValue = 3
        actionState.SAQs[3].QValue = 5
        action = self.qlearn.getActionForMaxQValue(actionState)
        self.assertTrue(action == None or action == 'right')
   


    if __name__ == '__main__':
        unittest.main()

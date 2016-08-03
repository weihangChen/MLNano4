import unittest
import pytest
#from smartcab.algorithm import QLearn
from smartcab.agent import *

class Test_MDPTest(unittest.TestCase):
    
    def setUp(self):
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.states = []
        
        for x in range(1,4):
            for y in range(1,4):
                location = Location(x, y)
                state = State(location)
                self.states.append(state)
                for action in self.valid_actions:
                    model = StateActionQValueModel(state, action)
                    state.SAQs.append(model)

        self.statepositivereward = next(x for x in self.states if x.Location.X == 3 and x.Location.Y == 1) 
        self.statepositivereward.Reward = 10      
        self.statenegativereward = next(x for x in self.states if x.Location.X == 3 and x.Location.Y == 2)
        self.statenegativereward.Reward = -10
   
    def test_Equal(self):
        #location
        l1 = Location(1,2)
        l2 = Location(1,2)
        l3 = Location(1,3)
        self.assertTrue(l1 == l2)
        self.assertTrue(l1 != l3)
        
        #state
        state1 = State(l1)
        state2 = State(l2)
        state3 = State(l3)
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
        #fake some qvalue for [x:3, y:1] and [x:2, y:1]
        self.statepositivereward.SAQs[0].QValue = 5
        self.statepositivereward.SAQs[1].QValue = 4
        self.statepositivereward.SAQs[2].QValue = 3
        
        previousState = next(x for x in self.states if x.Location.X == 2 and x.Location.Y == 1) 
        saqprevious = next(x for x in previousState.SAQs if x.Action == "right")
        qlearn = QLearn(epsilon=0.1, alpha=0.1, gamma=0.5)
        
        #case1
        qlearn.backwardPropagationQValueFromCurrentStateToPreviousState(saqprevious, self.statepositivereward.Reward, self.statepositivereward)
        self.assertTrue(saqprevious.QValue == 10)
        #case2
        saqprevious.QValue = 3
        qlearn.backwardPropagationQValueFromCurrentStateToPreviousState(saqprevious, self.statepositivereward.Reward, self.statepositivereward)
        self.assertTrue(saqprevious.QValue == 3.95)




        #qpreviousstate = 3
        #maxq = max([x.QValue for x in self.statepositivereward.SAQs])
        #reward = self.statepositivereward.Reward
        ##calculate
        #backwardpropagationvalue = qpreviousstate + qlearn.alpha * (reward +
        #qlearn.gamma * maxq - qpreviousstate)
        ## verify the backwardpropagation value
        #self.assertTrue(backwardpropagationvalue == 3.95)
    
    def test_ActionChoose(self):
        #fake some qvalue for [x:3, y:1] and [x:2, y:1]
        self.statepositivereward.SAQs[0].QValue = 5
        self.statepositivereward.SAQs[1].QValue = 4
        self.statepositivereward.SAQs[2].QValue = 3
        self.statepositivereward.SAQs[3].QValue = 5
        qlearn = QLearn(epsilon=0.1, alpha=0.1, gamma=0.5)
        action = qlearn.getActionForMaxQValue(self.statepositivereward)
        self.assertTrue(action == None or action == 'right')
   


    if __name__ == '__main__':
        unittest.main()

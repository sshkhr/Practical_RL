from collections import defaultdict
import random, math
import numpy as np

class EVSarsaAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Expected Value SARSA Agent.

        The two main methods are 
        - self.getAction(state) - returns agent's action in that state
        - self.update(state,action,nextState,reward) - returns agent's next action

        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self,state,action,value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """ 
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}
          
        Hint: all other methods from QLearningAgent are still accessible.
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        
        #<YOUR CODE HERE: SEE DOCSTRING>
        possible_values = [self.get_qvalue(state,action) for action in possible_actions]
        index = np.argmax(possible_values)
        state_value = epsilon * possible_values[index] + (1 - epsilon)*(np.sum(possible_values))/len(possible_actions)
        
        return state_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        #agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #<YOUR CODE HERE>
        q_value = (1-learning_rate)*self.get_qvalue(state,action) + learning_rate*(reward + gamma*self.get_value(next_state))
        
        self.set_qvalue(state, action, q_value)

    
    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_q_values = [self.get_qvalue(state,action) for action in possible_actions]
        index = np.argmax(possible_q_values)
        best_action =  possible_actions[index]

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).
        
        Note: To pick randomly from a list, use random.choice(list). 
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #agent parameters:
        epsilon = self.epsilon
        self.epsilon = 0.99*epsilon

        #<YOUR CODE HERE>
        choice = np.random.random() > epsilon
        
        if choice:
            chosen_action = self.get_best_action(state)
        else:
            chosen_action = random.choice(possible_actions)
        
        return chosen_action
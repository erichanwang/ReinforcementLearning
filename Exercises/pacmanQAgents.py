
from Exercises.qlearningAgents import QLearningAgent
import Helpers.util
from Helpers.featureExtractors import *



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = Helpers.util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = Helpers.util.Counter()

    def getWeights(self):
        return self.weights
    def getQValue(self, state, action):
        """
        Returns Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += self.weights[feature] * value
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Updates weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        correction = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * correction * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs:np.ndarray):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # return the action that maxinmizes the Q-value 
        # at the current observation as the output
        action = np.argmax(self.critic.qa_values(observation), axis=1)
        # TODO: check that the action type/dimension is as desired. 
        return action.squeeze()
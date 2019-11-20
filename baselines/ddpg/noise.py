import numpy as np

class AdaptiveParamNoiseSpec(object):
    """
    Implements adaptive parameter noise as seen in 'Parameter Space Noise For Exploration' by
    Plappert et al.

    :param initial_stddev: (float) the initial value for the standard deviation of the noise
    :param desired_action_stddev: (float) the desired value for the standard deviation of the noise
    :param adoption_coefficient: (float) the update coefficient for the standard deviation of the noise
    """

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """
        update the standard deviation for the parameter noise

        :param distance: (float) the noise distance applied to the parameters
        """

        if distance > self.desired_action_stddev:
            # decrease stddev
            self.current_stddev /= self.adoption_coefficient
        else:
            # increase stddev
            self.current_stddev += self.adoption_coefficient

    def get_stats(self):
        """
        return the standard deviation for the parameter noise

        :return: (dict) the stats of the noise
        """
        return {'param_noise_stddev': self.current_stddev}

    def __repr__(self):
        """

        :return: (string) printout of variables used in Adaptive Parameter Noise function
        """
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

class ActionNoise(object):


class NormalActionNoise(ActionNoise):


class OrnsteinUhlenbeckActionNoise(ActionNoise):


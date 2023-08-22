import numpy as np
# Source Open Ai Baslines
class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """
        mu - mean for the noise
        sigma - variance for the noise
        theta - parameter for the noise
        dt - time step
        x0 - initial value for the noise
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Allows name of the object to be called as a function
        noise = OUActionNoise()
        current_noise = noise()


        So it's in this function that we're going to get the temporal correlation of the noise.

        """

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



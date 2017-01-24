import numpy as np
from numpy import sqrt, cos, sin, log, pi

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    def sample(self):
        # Box-Muller transformation
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
        return self.mu + self.sigma * z1
        
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu
        self.Sigma = Sigma
    def sample(self):
        # Multivariate Normal Distribution - Cholesky
        Z = []
        L = np.linalg.cholesky(self.Sigma)
        for i in range(0, self.Mu.size):
            Z.append(UnivariateNormal(0, 1).sample())
        Z = np.array(Z)
        return self.Mu + np.dot(L, Z)
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap
    def sample(self):
        sum = 0
        intervals = []
        for i in range(0, self.ap.size):
            sum += self.ap[i]
            intervals.append([sum - self.ap[i], sum])
        intervals = np.array(intervals)
        u = np.random.uniform(0, sum)
        for i in range(0, self.ap.size):
            if u >= intervals[i][0] and u < intervals[i][1]:
                return i
        return -1



# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm
    def sample(self):
        sum = 0
        intervals = []
        for i in range(0, self.ap.size):
            sum += self.ap[i]
            intervals.append([sum - self.ap[i], sum])
        intervals = np.array(intervals)
        u = np.random.uniform(0, sum)
        for i in range(0, self.ap.size):
            if u >= intervals[i][0] and u < intervals[i][1]:
                return self.pm[i].sample()
        return -1
        
def main():
    I = np.diag([1,1])
    models = (MultiVariateNormal(np.array([1,1]), I), MultiVariateNormal(np.array([-1,-1]), I), MultiVariateNormal(np.array([1,-1]), I), MultiVariateNormal(np.array([-1,1]), I))
    weights = np.array([0.25,0.25,0.25,0.25])
    n = 100
    count = 0.0
    for i in range(0, n):
        sample = MixtureModel(weights, models).sample()
        if np.linalg.norm(sample - np.array([0.1,0.2])) <= 1:
            count += 1
    print 'The probability that a sample from this mixture distribution lies within the unit circle centered at (0.1, 0.2) is', count / n, '.'

if __name__ == "__main__":
    main()
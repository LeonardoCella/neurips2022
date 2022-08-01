from numpy import zeros, transpose, sqrt, outer, matrix, log, eye, dot, asarray, argmax
from numpy.linalg import qr, norm, matrix_rank, det
from sklearn import linear_model
from featLearnBan.policies.Policy import Policy
from featLearnBan.tools.matrix_functions import sherman_morrison


class Greedy(Policy):
    def __init__(self, horizon, d, nb_arms, Bhat):
        self._T = horizon
        self._n_arms = nb_arms
        self._d = d
        self._Bhat = Bhat
        self._rhat = Bhat.shape[1]
        self._w_hat = zeros(self._rhat) 
        self._X = []
        self._Y = []
        self._t = 0
        # print("INIT Bhat {}".format(Bhat))


    def update(self, j, t, arm, reward):
        # Update the estimation based on the current representation Bhat
        self._Y.append(reward)
        # print("\nUPDATE()\n_X {},{}, _Bhat {}".format(len(self._X), self._X[0].shape, self._Bhat.shape))
        # print("\nX {}, Bhat {}".format(self._X, self._Bhat))
        Xprime = dot(self._X, self._Bhat)
        # print("\n Xprime {}, Y {}".format(Xprime, self._Y))
        ols_model = linear_model.Ridge(alpha = 0.01).fit(Xprime, self._Y)
        self._w_hat = ols_model.coef_
        

    def choice(self, j, t, context_vectors):
        # contexts : narms x d
        # Bhat: d x rhat
        # print("\nCHOICE\ncontext shape {}, Bhat {}".format(context_vectors.shape, self._Bhat.shape))
        Xprime = dot(context_vectors, self._Bhat)
        wtx_array = dot(Xprime, self._w_hat).T
        max_index = list(wtx_array).index(max(wtx_array))
        self._X.append(context_vectors[max_index])
        return max_index


    def reset(self):
        return


    def getEstimate(self):
        return self._w_hat

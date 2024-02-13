from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import RadiusNeighborsRegressor as RNC
import numpy as np

'''
learn distance metric on X. 
match tightly on X via caliper.
matching tightly on T via caliper inside each nearest neighbor set.
estimate conditional average dosage response function
'''

class cont_lcm():
    def __init__(self, caliper_X, caliper_T, T_grid):
        self.caliper_X = caliper_X
        self.caliper_T = caliper_T
        self.T_grid = T_grid

    def fit(self, X_tr, Y_tr, ridge_params = {}):
        '''
        run ridge regression 
        '''

        ridge_model = Ridge(**ridge_params)
        ridge_model.fit(X_tr, Y_tr)

        self.M = ridge_model.coef_

    def estimate_cadrf_at_t(self, X_nn_idxs, T_est, Y_est):
        '''
        1. isolate set of nns in X space
        2. of these NNs, find NNs in T space
        3. average Y of all NNs
        '''

        MG_Y = Y_est[X_nn_idxs]
        MG_T = T_est[X_nn_idxs]
        CADRF = []
        for T_query in self.T_grid:
            dist_T = np.abs(MG_T - T_query) 
            T_nn_idxs = np.where(dist_T <= self.caliper_T)[0]
            CADRF.predict(MG_Y[T_nn_idxs].mean())

        return np.array(CADRF)


    def estimate_cadrf(self, X_est, T_est, Y_est):
        '''
        match on X using learned distance metric
        match on T within each X matched group
        '''
        # scale covariates
        X_est = X_est * self.M

        X_NN_model = NN(radius = self.caliper_X, algorithm = 'ball_tree')
        X_NN_model.fit(X_est)

        match_groups = X_NN_model.radius_neighbors(X_est, radius = self.caliper_X, return_distance=False)

        cadrfs = np.apply_along_axis(func1d = lambda nn_i : self.estimate_cadrf_at_t(nn_i, T_est, Y_est),
                                     axis = 0,
                                     arr = match_groups)
        
        return cadrfs
    
    def estimate_adrf(self, cadrfs = None, X_est = None, T_est = None, Y_est = None):
        if cadrfs == None:
            cadrfs = self.estimate_cadrf(X_est, T_est, Y_est)
        
        return cadrfs.mean(axis = 0)
    
    def fit_estimate(self, X_tr, Y_tr, X_est, T_est, Y_est, ridge_params = {}):
        self.fit(X_tr, Y_tr, ridge_params)

        return self.estimate_cadrf(X_est, T_est, Y_est)
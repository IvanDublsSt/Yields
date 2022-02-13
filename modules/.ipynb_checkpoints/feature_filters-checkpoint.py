import pandas as pd
import numpy as np
import modules.missing_values_module as mvm
import modules.cliques_filter_module as cliques 

class SeparatedDF():
    def __init__(self, X, categorical_variables = [], ignore_dummies = False):
        if (categorical_variables == [])&(ignore_dummies == True):
            categorical_variables = find_dummies(X)
        self.X_numeric = X.drop(categorical_variables, 1)
        self.X_categorical = X.copy()[categorical_variables]
        

def find_dummies(X):
    count_values = X.apply(lambda x: len(x.unique()), axis = 0)
    dummy_mask = (X.apply(lambda x: len(x.unique()), axis = 0) <= 2)
    dummy_columns = count_values[dummy_mask].index.tolist()
    return dummy_columns


class OutputCorrelationFilter():


    def __init__(
                self, n_features = False, 
                share_features = False,  
                max_acceptable_correlation = False,
                corr_metrics = "pearson",
                categorical_variables = []
                ):
        
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() > 1:
            raise ValueError("Please, predefine ONE OF: acceptable correlation, share of features or number of features")
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() == 0:
            raise ValueError("Please, choose either acceptable correlation, share of features or number of features")
            
        self.n_features = n_features
        self.share_features = share_features
        self.corr_metrics = corr_metrics
        self.max_acceptable_correlation = max_acceptable_correlation
        self.categorical_variables = categorical_variables
        
    def fit(self, X, y):
#         print("I am fitting Ofilter")
        
        X_separated = SeparatedDF(X, self.categorical_variables, ignore_dummies = True)
        if self.share_features != False:
            self.n_features = round(len(X_separated.X_numeric.columns) * share_features)
        if self.n_features == 0:
            self.n_features = 1
        corrs = X_separated.X_numeric.apply( lambda column: column.corr(y, method = self.corr_metrics) )
        if self.max_acceptable_correlation != False:
            self.n_features = (corrs <= self.max_acceptable_correlation).sum()
        self.desired_names = corrs.sort_values(ascending = False).index[:self.n_features].tolist()
#         print("Desired names from filter:")
#         print(self.desired_names)
        self.dummy_names = find_dummies(X)
        return self
    
    def transform(self, X, y= None):
        
        X_reunited = pd.concat([X[self.desired_names], X[find_dummies(X)]], axis = 1)

        return X_reunited

    
class MutualCorrelationFilter():
    
    def __init__(
                self, n_features = False, 
                share_features = False, 
                max_acceptable_correlation = False,
                corr_metrics = "pearson",
                categorical_variables = []
                ):
        
        
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() > 1:
            raise ValueError("Please, predefine ONE OF: acceptable correlation, share of features or number of features")
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() == 0:
            raise ValueError("Please, choose either acceptable correlation, share of features or number of features")
            
        self.n_features = n_features
        self.share_features = share_features
        self.corr_metrics = corr_metrics
        self.max_acceptable_correlation = max_acceptable_correlation
        self.categorical_variables = categorical_variables

    def fit(self, X, y = None):
        X_separated = SeparatedDF(X, self.categorical_variables, ignore_dummies = True)
        if self.share_features != False:
            self.n_features = round(len(X.columns) * self.share_features)
        
        if (self.n_features == 0)&(self.share_features != False):
            self.n_features = 1
        
        current_df = X_separated.X_numeric.copy()
        while len(current_df.columns) > self.n_features:
            current_corr = current_df.corr(method = self.corr_metrics).unstack().sort_values(ascending = False)
            for i in range(len(current_corr)):
                column_names = current_corr.index[i]
                corr_value = current_corr.iloc[i]
                if column_names[0] != column_names[1]:
                    most_correlated_column = column_names[0]
                    break
                    
            if len(current_df.columns) == 1:
                break
                
            if corr_value <= self.max_acceptable_correlation:
                break
                
            current_df.drop(most_correlated_column, axis = 1, inplace = True)
        
        self.desired_names = current_df.columns.tolist()
        self.dummies_found = find_dummies(X)
        return self
    
    def transform(self, X, y= None):
        X_reunited = pd.concat([X[self.desired_names], X[self.dummies_found]], axis = 1)

        return X_reunited
        
class VIFFilter():
    
    def __init__(self, acceptable_vif = [5, 10]):
        
        self.acceptable_vif = acceptable_vif
        
    def fit(self, X, y = None):
        vifs = pd.Series(np.linalg.inv(X.corr().to_numpy()).diagonal(), 
                 index=X.columns, 
                 name='VIF')
        self.desired_names = vifs[(vifs <= self.acceptable_vif[0]) | (vifs >= self.acceptable_vif[1])].index.tolist()
        return self
    
    def transform(self, X, y = None):
        
        return X[self.desired_names]

    
class CliquesFilter():
    
    def __init__(self, trsh = 0.65):
        self.trsh = 0.65
        
    def fit(self, X, y):
        qlq_list, G = cliques.get_noncollinear_fts(
                                                    X, y, trsh=0.65, mode="all", verbose=False
                                                    )
        self.desired_names = qlq_list[list(qlq_list)[0]][0]
        
    def transform(self, X, y = None):
        
        return X[self.desired_names]

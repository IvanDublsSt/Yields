
#modules
import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import umap

from sklearn import datasets, metrics, model_selection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline

from hyperopt import hp
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# import umap # does not work for me, since I installed umap instaed of umap-learn by mistake
import umap.umap_ as umap

# for HyperOpt class
import lightgbm as lgb
import xgboost as xgb
# import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

# новый пакет!
from feature_engine.encoding import WoEEncoder
from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.encoding import OneHotEncoder

from typing import List, Union
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _check_input_parameter_variables

from mlxtend.feature_selection import SequentialFeatureSelector
from feature_engine.selection  import SelectByShuffling
from feature_engine.selection  import RecursiveFeatureAddition
from feature_engine.selection  import SmartCorrelatedSelection


#functions 

def Gini(y, y_pred):
    res = roc_auc_score(y, y_pred) * 2 - 1
    print(f"Gini: {res}")
    return(res)

def filter_params(params, pipe):
    '''
    From all input parameters filter only
    those that are relevant for the current
    pipeline
    '''
    pipe_steps = list(pipe.named_steps.keys())
    params_keys = list(params.keys())
    
    return {
        key: params[key]
        for key in params_keys
        if key.split('__')[0] in pipe_steps
    }

def construct_pipe(steps_dict, modules):
    '''
    Construct a pipeline given structure
    '''
    return [(steps_dict[s], modules[steps_dict[s]]) for s in steps_dict if steps_dict[s] != 'skip']


class PipeHPOpt(object):
    '''
    Класс PipeHPOpt — Pipeline with hyperparameter optimisation
    using hyperopt — нацелен на оптимизацию пайплайна как с точки
    зрения входящих в него модулей, так и гиперпараметров каждого
    из модулей
    '''  
    def __init__(self, X, y, modules, pipe_para, task = "classification", mode='kfold', n_folds = 5, test_size=.33, seed=42, kfold_variable = None):
        '''   
        _inputs:
        X — train dataset
        y — train targets
        modules — dict of all modules that might potentially be included into
            the pipeline
        mode — wither "kfold" or "valid" (error if other) — sets if X, y will
            be subdivided into k cross-validation samples or train/test samples,
            respectively. "kfold" is default. Key advantage of valid: it returns
            the optimal model; in "kfold" mode the model should be retrained with
            optimal hyperparameters
        n_folds — number of folds at cross-validation (5 is default). Applied
            only if mode = "kfold". Warning added
        test_size — test sample % (.33 is default). Applied only if mode = "valid". 
            Warning added
        seed — random seed (42 is default)
        '''
#         breakpoint()
        if (mode != 'kfold') & (mode != 'valid') & (mode != 'kfold_variable'):
            raise ValueError("Choose mode 'kfold' or 'valid' or 'kfold_variable'")
        if (mode == 'valid') & (n_folds != 5):
            import warnings
            warnings.warn("Non-default n_folds won't be used since mode == valid!")
        if ((mode == 'kfold') or (mode == 'kfold_variable')) & (test_size != .33):
            import warnings
            warnings.warn("Non-default test_size won't be used since mode == kfold or kfold_variable!")
        if (mode == "kfold_variable")&(kfold_variable == None):
            import warnings
            warnings.warn("You chose kfold_variable method but provided no variable to sample! Using kfold instead!")
            mode = "kfold"

        self.X       = X
        self.y       = y
        self.mode    = mode
        self.n_folds = n_folds
        self.seed    = seed
        self.modules = modules
        self.kfold_variable = kfold_variable
        self.pipe_para = pipe_para
        self.task = task
        if mode == 'valid':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
#         breakpoint()

    def process(self, space, trials, algo, max_evals, fn_name='_pipe'):
        '''
        _inputs: TBD
        
        _output:
        result: hyperopt weird object of the optimal model representation
        trials: info on each of the hyperopt trials
        '''
#         breakpoint()
#         print(para)
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        self.result = result
        self.trials = trials
#         breakpoint()

        return result, trials

    
    def get_best_params(self):
#         breakpoint()
        return self.trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
    
    def get_best_model(self):
#         breakpoint()
        para = self.get_best_params()
        pipe_steps = [(para['pipe_params'][i], self.modules[para['pipe_params'][i]]) for i in para['pipe_params'] if para['pipe_params'][i] != 'skip']
        reg = Pipeline(pipe_steps)
        for p in self.pipe_para['set_params']:
            try:
                reg.set_params({p: para[p]})
            except:
                pass # repetition, not DRY, think how to delete
        breakpoint()
        return reg.fit(self.X, self.y)
    
    def _pipe(self, para):
#         breakpoint()
        # print(para)
        pipe_steps = [(para['pipe_params'][i], self.modules[para['pipe_params'][i]]) for i in para['pipe_params'] if para['pipe_params'][i] != 'skip']
        reg = Pipeline(pipe_steps)
        print(reg)
        for p in self.pipe_para['set_params']:
            try:
                reg.set_params({p: para[p]})
            except:
                pass
#         breakpoint()
        if self.mode == 'kfold':
            return self._train_reg_kfold(reg, para)
        elif self.mode == 'valid':
            return self._train_reg_valid(reg, para)

    def _train_reg_valid(self, reg, para):
#         breakpoint()
        reg.fit(self.x_train, self.y_train)
        if self.task == "classification":
            pred = reg.predict_proba(self.x_test)[:, 1]
        elif self.task == "regression":
            pred = reg.predict(self.x_test)[:, 1]
        pred = reg.predict_proba(self.x_test)[:, 1]
        loss = para['loss_func'](self.y_test, pred)
#         breakpoint()
        return {'loss': loss, 'model': reg, 'params': para, 'status': STATUS_OK}
    
    def _train_reg_kfold(self, reg, para):
#         breakpoint()
        if self.mode == "kfold":
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            losses = []
            for train_index, test_index in kf.split(self.X):
                X_split_train, X_split_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
                y_split_train, y_split_test = self.y.iloc[train_index, ],  self.y.iloc[test_index, ]
                reg.fit(X_split_train, y_split_train)
                if self.task == "classification":
                    pred = reg.predict_proba(X_split_test)[:, 1]
                elif self.task == "regression":
                    pred = reg.predict(X_split_test)[:, 1]
                loss = para['loss_func'](y_split_test, pred)
                losses.append(loss)
            return {'loss': np.mean(losses), 'params': para, 'status': STATUS_OK}
        elif self.mode == "kfold_variable":
            kf = GroupKFold(n_splits=5)
            losses = []
            for train_index, test_index in kf.split(self.X, groups = self.X[self.kfold_variable]):
                X_split_train, X_split_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
                y_split_train, y_split_test = self.y.iloc[train_index, ],  self.y.iloc[test_index, ]
                reg.fit(X_split_train, y_split_train)
                if self.task == "classification":
                    pred = reg.predict_proba(X_split_test)[:, 1]
                elif self.task == "regression":
                    pred = reg.predict(X_split_test)[:, 1]                
                loss = para['loss_func'](y_split_test, pred)
                losses.append(loss)
            return {'loss': np.mean(losses), 'params': para, 'status': STATUS_OK}            

    
    
#adjusted modules

class CombineWithReferenceFeature_adj():
    """
    Обертка вокруг CombineWithReferenceFeature()
    Позволяет не устанавливать параметры
    + variables_to_combine
    + reference_variables
    заранее (иначе не будет работать с OneHotEncoder
    и прочими преобразователями данных, а делать это при .fit()
    """
    def __init__(self, operations):
        self.operations = operations
        
    def fit(self, X, y):
        self.combinator = CombineWithReferenceFeature(
            variables_to_combine = list(X.columns),
            reference_variables = list(X.columns),
            operations = self.operations
        )
        self.combinator.fit(X, y)
        return(self)
    
    def transform(self, X):
        return(self.combinator.transform(X))        

class KernelPCA_adj():
    """
    Обертка нужна, чтобы transform() возвращал
    pd.df(), а не np.array() - не все могут в np
    """
    def __init__(self, **kwargs):
        self.kpca = sklearn.decomposition.KernelPCA(**kwargs)
        
    def fit(self, X, y):
        self.kpca.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.kpca.transform(X), index = X.index)], axis=1)
    
    def set_params(self, **kwargs):
        self.kpca.set_params(**kwargs)
        return self 

class PCA_adj():
    """
    Обертка нужна, чтобы transform() возвращал
    pd.df(), а не np.array() - не все могут в np
    """
    def __init__(self, **kwargs):
        self.pca = sklearn.decomposition.PCA(**kwargs)
        
    def fit(self, X, y):
        self.pca.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.pca.transform(X), index = X.index)], axis=1)
    
    def set_params(self, **kwargs):
        self.pca.set_params(**kwargs)
        return self 

class Isomap_adj():
    """
    Обертка нужна, чтобы transform() возвращал
    pd.df(), а не np.array() - не все могут в np
    """
    def __init__(self, **kwargs):
        self.isomap = sklearn.manifold.Isomap(**kwargs)
        
    def fit(self, X, y):
        self.isomap.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.isomap.transform(X), index = X.index)], axis=1)
    
    def set_params(self, **kwargs):
        self.isomap.set_params(**kwargs)
        return self 

class UMAP_adj():
    """
    Обертка нужна, чтобы transform() возвращал
    pd.df(), а не np.array() - не все могут в np
    """
    def __init__(self, **kwargs):
        self.umap = umap.UMAP(**kwargs)
        
    def fit(self, X, y):
        self.umap.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.umap.transform(X), index = X.index)], axis=1)
    
    def set_params(self, **kwargs):
        self.umap.set_params(**kwargs)
        return self 
    
    

class WoEEncoder_adj(BaseCategoricalTransformer):
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")

        self.variables = _check_input_parameter_variables(variables)
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the WoE.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.
        y: pandas series.
            Target, must be binary.
        """
        
        X = self._check_fit_input_and_variables(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # check that y is binary
        if y.nunique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        if any(x for x in y.unique() if x not in [0, 1]):
            temp["target"] = np.where(temp["target"] == y.unique()[0], 0, 1)

        self.encoder_dict_ = {}

        total_pos = temp["target"].sum()
        total_neg = len(temp) - total_pos
        temp["non_target"] = np.where(temp["target"] == 1, 0, 1)

        for var in self.variables_:
            pos = (temp.groupby([var])["target"].sum() + .5) / total_pos
            neg = (temp.groupby([var])["non_target"].sum() + .5) / total_neg

            t = pd.concat([pos, neg], axis=1)
            t["woe"] = np.log(t["target"] / t["non_target"])

            # we make an adjustment to override this error
            # if (
            #     not t.loc[t["target"] == 0, :].empty
            #     or not t.loc[t["non_target"] == 0, :].empty
            # ):
            #     raise ValueError(
            #         "The proportion of one of the classes for a category in "
            #         "variable {} is zero, and log of zero is not defined".format(var)
            #     )

            self.encoder_dict_[var] = t["woe"].to_dict()

        self._check_encoding_dictionary()

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().inverse_transform(X)

        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__

    def _more_tags(self):
        tags_dict = _return_tags()
        # in the current format, the tests are performed using continuous np.arrays
        # this means that when we encode some of the values, the denominator is 0
        # and this the transformer raises an error, and the test fails.
        # For this reason, most sklearn transformers will fail. And it has nothing to
        # do with the class not being compatible, it is just that the inputs passed
        # are not suitable
        tags_dict["_skip_test"] = True
        return tags_dict
    
    
def optimizer(X_train, y_train, pipe_para, modules, mode = "kfold", kfold_variable = None):
    hpoptimizer = PipeHPOpt(X_train, y_train, modules=modules, mode='kfold', n_folds = 5, seed=42, kfold_variable = None)
    lgb_opt, trials = hpoptimizer.process(space=pipe_para, trials=Trials(), algo=tpe.suggest, max_evals=10)
    return hpoptimizer

def algorithm_hint(hint = True):
    print("1. Define modules separately (separate_modules_hint)")
    separate_modules_hint(hint)
    print("2. Define modules dictionary (dictionary_modules_hint)")
    dictionary_modules_hint(hint)
    print("3. Define parameters pipeline (pipeline_modules_hint)")
    pipeline_modules_hint(hint)
    print("4. Define hyperparameters dictionary (hyper_hint)")
    hyper_hint(hint)
    print("5. Define loss function")
    print("6. Gather modules, parameters and loss into a dictionary (use gather_function)")
    print("7. Run optimizer")
    
def separate_modules_hint(hint = True):
    print("\n<variable for module i> = <module function i> for i in modules\n" * hint)

def dictionary_modules_hint(hint = True):
    print("\n<dictionary name> = {'<module name string i>' :\n <variable for module i> for i in modules}\n" * hint)

def pipeline_modules_hint(hint = True):
    print("\n<pipeline modules variable name> = {'<pipeline stage name j>' :\n hp.choice('<pipeline stage name j>', ['<module name string i>' for i in modules of stage j]) for j in stages}\n'skip' as '<module name string i>' if you want to try doing nothing on the stage\n" * hint)
    
def hyper_hint(hint = True):
    print("\n<modules hyperparameters list variable name> = {'<module name string i>__<parameter k name>' :\n [<parameter k value option m> for m in a list of possible values] for k in hyperparameters of module i, for i in modules}\n" * hint)
    
def gather_function(pipe_params, set_params, loss_func = lambda y, pred: -sklearn.metrics.roc_auc_score(y, pred)):
    pipe_para = dict()
    pipe_para['pipe_params']    = pipe_params
    pipe_para['set_params']     = set_params
    pipe_para['loss_func']      = loss_func 
    return pipe_para



    
#https://github.com/dmitryokhotnikov/Habr/blob/main/Article_experiments_code.ipynb
from datetime import timedelta
from time import perf_counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels
import statsmodels.formula.api as smf

def get_scores(dataset, target, rnd_state=1234):
    """Returns features scores.

    Parameters
    ----------
    dataset: pd.DataFrame. Dataset with all data.
    target: array. Ground truth of target.
    rnd_state: integer. Random state.

    Returns
    -------
    scores: dictionary. Features and their scores.
    """

    scores = dict()

    for feature in dataset.columns:
        mask = ~dataset[feature].isna()
        X = dataset[mask][[feature]]
        y = target[mask]
        discr_fts_mask = X.dtypes.values == 'int64'

        score = mutual_info_classif(
            X=X,
            y=y,
            discrete_features=discr_fts_mask,
            n_neighbors=5,
            random_state=rnd_state
        )
        scores[feature] = score[0]

    return scores
def get_noncollinear_fts(dataset, target, trsh=0.8, mode="all", random_seed=1234, verbose=True):
    """Returns maximum linearly independent subset of features by threshold.

    Parameters
    ----------
    dataset: pd.DataFrame. Dataset with all data.
    target: array. Ground truth of target.
    trsh: float. threshold of correlation.  
    mode: string. 'all' returns all sets 'max' returns max set
    random_seed: integer. Random state.
    verbose: bool. Print steps or not

    Returns
    -------
    answer: dictionary. key - length of set, value - features and total score
    G: graph
    """

    t0 = perf_counter()

    # 1. MI calculation
    if verbose:
        print(f"=> mutual info calculation...")
    t1 = perf_counter()
    scores = get_scores(dataset, target, random_seed)
    if verbose:
        print(f"Task completed in: {timedelta(seconds=(perf_counter()-t1))}\n")

    # 2. Correlation matrix calculation
    if verbose:
        print(f"=> corr_matrix calculation...")
    t1 = perf_counter()
    corr_matrix = dataset.corr().abs()
    fts = corr_matrix.columns
    corr_matrix = np.array(corr_matrix)
    if verbose:
        print(f"Task completed in: {timedelta(seconds=(perf_counter()-t1))}\n")

    # 3. Graph assembling
    if verbose:
        print(f"=> graph assembling...")
    t1 = perf_counter()
    # Fill diagonal elements by 2. That value is greater than trashold
    np.fill_diagonal(corr_matrix, 2)
    corr_matrix = pd.DataFrame(corr_matrix, columns=fts, index=fts)
    graph_matrix = corr_matrix[abs(corr_matrix) > trsh]
    # High-correlated vertices does not connect by edge
    graph_matrix[~graph_matrix.isna()] = 0
    # Other vertices connect by edges
    graph_matrix.fillna(1, inplace=True)
    G = nx.from_numpy_matrix(np.array(graph_matrix))
    G = nx.relabel_nodes(G, dict(zip(list(G.nodes), fts)))
    if verbose:
        print(f"Task completed in: {timedelta(seconds=(perf_counter()-t1))}\n")

    # 4. Qliques search
    if verbose:
        print(f"=> qliques search...")
    t1 = perf_counter()
    clq = nx.find_cliques(G)
    cliques = list(clq)
    if verbose:
        print(f"{len(cliques)} qliques are found:\n")
    lens = np.array(list(map(len, cliques)))  # size of cliques
    stat = pd.Series(lens).value_counts()  # qliques distribution by size
    if verbose:
        print(stat)
    if verbose:
        print(f"Task completed in: {timedelta(seconds=(perf_counter()-t1))}\n")

    # 5. Optimal qlique search
    if verbose:
        print(f"=> best cliques search...")
    t1 = perf_counter()
    answer = {}
    if mode == "all":
        iter_list = sorted(np.array(stat.index))
    elif mode == "max":
        iter_list = [np.array(stat.index).max()]
    for val in iter_list:
        if verbose:
            print(f"Search best clique for dim = {val}")
        max_curr = -1e5
        fts_list = []
        for idx in np.argwhere(lens == val).ravel():
            summa = 0
            for col in cliques[idx]:
                summa += scores[col]
            if summa > max_curr:
                max_curr = summa
                fts_list = cliques[idx]
        answer[val] = (fts_list, max_curr)
    if verbose:
        print(f"Task completed in: {timedelta(seconds=(perf_counter()-t1))}\n")
    print(f"All tasks completed in: {timedelta(seconds=(perf_counter()-t0))}")

    return answer, G
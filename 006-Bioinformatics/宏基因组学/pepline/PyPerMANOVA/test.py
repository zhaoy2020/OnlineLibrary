from PyPerMANOVA import permutational_analysis

import sklearn.datasets
import seaborn as sns
import pandas as pd

def make_groups(n_clusters = 3,
                d_differences = [2,-2],
                n_features = 4,
                samples_per_cluster = 20,
                random_state = 42,
                draw=True,
                **kwargs
                ):

    first_center = tuple(0 for i in range(n_features))
    centers = [first_center] + [tuple(x+d for x in first_center) for d in d_differences]
    centers = centers[:n_clusters] 
    
    raw_data = sklearn.datasets.make_blobs(n_samples=samples_per_cluster*n_clusters,
                                           n_features=n_features,
                                           centers=centers,
                                           random_state = random_state,
                                           **kwargs
                                           )
    
    raw_df = pd.DataFrame(data  = raw_data[0],
                          index = raw_data[1],
                          columns = [f"feature_{i+1}" for i in range(raw_data[0].shape[1])])
    raw_df.index = raw_df.index.astype(str)
    if draw:
        #draw dataframe
        sns.scatterplot(data=raw_df,x="feature_1",y="feature_2",hue=raw_df.index)
        
    return(raw_df)
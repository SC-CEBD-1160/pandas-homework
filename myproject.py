from sklearn.datasets import load_wine
import pandas as pd

def pretty_print(name, to_print):
    print(f'{name}:')
    print(f'{to_print}\n\n')

data = load_wine()

wine = pd.DataFrame(data=data['data'],
                    columns= data['feature_names'])

print(wine.to_string)
# compute and print information and summary statistics on the dataset

pretty_print("Show all column names for a dataframe", wine.columns)
pretty_print("Getting the shape of a dataframe", wine.shape)
pretty_print("Summarized info on dataframe", wine.info())
pretty_print("Quick stats on all numeric columns for dataframe", wine.describe())

# pretty_print("Selecting only the alcalinity_of_ash column", wine['alcalinity_of_ash'])

# compute and print correlations on the dataset

pretty_print("print correlations on the dataframe", wine.corr())
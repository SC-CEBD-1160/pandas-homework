import numpy as np
import pandas as pd

def pretty_print(name, to_print):
    print(f'{name}:')
    print(f'{to_print}\n\n')

# Creating Dataframe from Files
insurance = pd.read_csv(filepath_or_buffer='data/insurance.csv',
                      sep=',',
                      header=0)

pretty_print("insurance dataframe", insurance.to_string())
pretty_print("insurance columns", insurance.columns)
pretty_print("insurance columns", insurance.index)
pretty_print("insurance dtypes", insurance.dtypes)
pretty_print("insurance shape", insurance.shape)
pretty_print("insurance info", insurance.info())
pretty_print("insurance describe", insurance.describe())

pretty_print("selecting the age column", insurance['age'])
pretty_print("selecting the age, children, charges column", insurance[['age', 'children', 'charges']])
pretty_print("selecting the age, children, charges column 5 first entries", insurance[['age', 'children', 'charges']].head(5))

pretty_print("selecting the min value", insurance['charges'].min())
pretty_print("selecting the max value", insurance['charges'].max())
pretty_print("selecting the avg value", insurance['charges'].mean())

pretty_print("selecting rows by multiple criteria",
             insurance[(insurance['charges'] == 10797.3362)][['age', 'sex']])

# is a smoker ?
if insurance[(insurance['charges'] == 10797.3362)]['smoker'].items() == 'yes':
    print('Is a smoker\n\n')
else:
    print('is not a smoker\n\n')

# age of maximum charge
pretty_print("age of maximum charge", insurance[(insurance['charges'] == insurance['charges'].max())]['age'].values[0])

# How many insured people do we have for each region
pretty_print("insured by region", insurance['region'].value_counts())

# How many insured people are children
pretty_print("how many insured children", insurance.dropna()['children'].count())

# What do you expect to be the correlation between
# children should have a correlation with charges, bmi and age
# There should be a correlation between age and charges

# correlation
pretty_print("correlation", insurance.corr()[['charges', 'age', 'bmi', 'children']])
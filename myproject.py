from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import os

def pretty_print(name, to_print):
    print(f'{name}:')
    print(f'{to_print}\n\n')

data = load_wine()
# print(data)

wine = pd.DataFrame(data=data['data'],
                    columns= data['feature_names'])

print(wine.to_string)
# compute and print information and summary statistics on the dataset

pretty_print("Show all column names for a dataframe", wine.columns)
pretty_print("Getting the shape of a dataframe", wine.shape)
pretty_print("Summarized info on dataframe", wine.info())
pretty_print("Quick stats on all numeric columns for dataframe", wine.describe())

pretty_print("Selecting only the alcalinity_of_ash column", wine['alcalinity_of_ash'])

# compute and print correlations on the dataset
pretty_print("print correlations on the dataframe", wine.corr())

# Plotting line chart
os.makedirs('plots', exist_ok=True)

plt.plot(wine['alcohol'], color='blue')
plt.title('Alcohol by Index')
plt.xlabel('Index')
plt.ylabel('Alcohol')
plt.savefig(f'plots/alcohol_by_index_plot.png', format='png')
plt.clf()

# Plotting histogram
plt.hist(wine['flavanoids'], bins=3, color='g')
plt.title('Flavanoids')
plt.xlabel('Flavanoids')
plt.ylabel('Count')
plt.savefig(f'plots/flavanoids_hist.png', format='png')
plt.clf()

# Plotting scatterplot
plt.scatter(wine['hue'], wine['color_intensity'], color='b')
plt.title('Hue to Color Intensity')
plt.xlabel('Hue')
plt.ylabel('Color Intensity')
plt.savefig(f'plots/hue_to_color_intensity.png', format='png')

plt.close()
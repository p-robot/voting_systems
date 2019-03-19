#!/usr/bin/env python3
"""
Plot the distribution of A) outbreak duration and B) total cattle culled for each model for the
data from the "objectives matter" paper.  This is just a sanity check of the data.  

This assumes there is data called 'model_a.csv' ... 'model_e.csv' in the 'data' folder (pull this from the github.com/p-robot/objectives_matter.git repo if not) and it also assumes there is a folder called 'graphics' in which figures can be saved.  

W. Probert, 2019
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from os.path import join

if __name__ == "__main__":
    datasets = ['model_a.csv', 'model_c.csv', 'model_d.csv', 'model_e.csv']
    
    results = []; all_durations = []; all_cattle_culled = []
    for dataset in datasets:
        df = pd.read_csv(join("data", dataset))
        results.append(df)
    
    # Set colours (be specific)
    bins = np.linspace(0, 1000, 200)
    fig, ax = plt.subplots(ncols = 2)
    for i, m in enumerate(['A', 'C', 'D', 'E']):
        ax[0].hist(df[df.model == m].duration, label = "Model "+m, 
            bins = bins, alpha = 0.4)
        ax[0].set_xlabel("Outbreak duration (days)")
        ax[1].hist(df[df.model == m].cattle_culled, label = "Model "+m, 
            bins = 100, alpha = 0.4)
        ax[1].set_xlabel("Cattle culled (head)")
    plt.legend(loc='upper right')
    plt.savefig(join("graphics", "output_measures.pdf"))
    plt.close()
    
    colors = ["#a6611a", "#dfc27d", "#80cdc1", "#018571"]
    
    fig, ax = plt.subplots(ncols = 2)
    for i, m in enumerate(['A', 'C', 'D', 'E']):
        
        data = df[df.model == m].duration.values
        density = stats.kde.gaussian_kde(data)
        
        x = np.linspace(0, 200, 100)
        ax[0].plot(x, density(x), label = "Model "+m, linewidth = 2, color = colors[i])
        ax[0].set_xlabel("Outbreak duration (days)")
        
        data = df[df.model == m].cattle_culled.values
        density = stats.kde.gaussian_kde(data)
        
        x = np.linspace(0, 30000, 100)
        ax[1].plot(x, density(x), label = "Model "+m, linewidth = 2, color = colors[i])
        ax[1].set_xlabel("Cattle culled (head)")
    plt.legend(loc='upper right')
    fig.set_size_inches(10, 5)
    plt.savefig(join("graphics", "output_measures_density.pdf"))
    plt.close()
    
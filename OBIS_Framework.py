import pickle
import Packages

# All processing functions are in Packages.py, this file only executes them and saves the results to a pickle file
# this pickle file holds the output of the preprocessing, such as the DF with LOF scores.
# From there, I used the Jupyter notebook called "LSTM.ipynb" to test the results, as it is easier with visualisations.
# Make sure the dataset is in the data folder (you can download it from google drive)
# In the config file you can further change the trajectories or designated folders.


if __name__ == "__main__":
    all_results = Packages.create_historical_outlier_dfs()
    pickle.dump(all_results, open("./results/OBIS_results.pickle", "wb"))

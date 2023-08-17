"""
This script takes in input the path to the debug folder used to store all
the experiments, then loads all the .txt files containing the copmuted statistics
and compact the results into a single csv
"""
import os
import pandas as pd
from os.path import basename


def load_statistics_from_file(statistics_file_path):
    file1 = open(statistics_file_path, 'r')
    Lines = file1.readlines()

    statistics = {}
    for line in Lines:
        statistic_name, statistic_value = line.split(':')

        statistics[statistic_name] = statistic_value.replace('\n','')
    return statistics

if __name__ == "__main__":

    experiment_root = '/Users/gianlucadestefano/Desktop/TESI/Statistiche'

    experiments = {}

    for experiment_path in [f.path for f in os.scandir(experiment_root) if f.is_dir()]:

        experiment_name = basename(experiment_path).replace('Analysis-Samples','')

        execution_timestamps = sorted([f.name for f in os.scandir(experiment_path) if f.is_dir()])

        statistics_file_path = os.path.join(experiment_path, execution_timestamps[0], 'statistics.txt')

        statistics_dict = load_statistics_from_file(statistics_file_path)

        experiments[experiment_name] = statistics_dict

    df = pd.DataFrame(experiments)

    df.to_csv('./full_results.csv')
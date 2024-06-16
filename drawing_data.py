import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from datetime import datetime

download_dir = "results/data_transitions/"

divide_rate = [0.7, 0.1, 0.2]

def to_datetime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

def main(source_file):
    source = pd.read_csv(source_file)
    columns = source.columns

    date = [to_datetime(date) for date in source[columns[0]]]

    start_len = [0, int(len(date) * divide_rate[0]), int(len(date) * (divide_rate[0] + divide_rate[1]))]
    end_len = [start_len[1], start_len[2], len(date)]

    for i, column in enumerate(columns[1:]):
        print(column)

        file_name = str(i+1) + "_" + column.replace(" ", "_").replace("/", "_").replace(":", "_") + ".png"
        values = source[column].values

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(date[start_len[0]:end_len[0]], values[start_len[0]:end_len[0]], label="Train", marker='o', markersize=1, linestyle='None')
        ax.plot(date[start_len[2]:end_len[2]], values[start_len[2]:end_len[2]], label="Test", marker='o', markersize=1, linestyle='None')
        ax.plot(date[start_len[1]:end_len[1]], values[start_len[1]:end_len[1]], label="Vali", marker='o', markersize=1, linestyle='None')

        plt.xlabel(columns[0])
        plt.ylabel(column)

        plt.legend()
        plt.title(f"[{i+1}]{column}")
        
        plt.savefig(download_dir + file_name)
        plt.close()

if __name__ == '__main__':
    try:
        os.mkdir(download_dir)
    except:
        pass

    args = sys.argv[1:]
    main(args[0])
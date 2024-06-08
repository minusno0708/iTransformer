import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from datetime import datetime

download_dir = "results/data_transitions/"

def to_datetime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

def main(source_file):
    source = pd.read_csv(source_file)
    columns = source.columns

    date = [to_datetime(date) for date in source[columns[0]]]

    for column in columns[1:]:
        print(column)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(date, source[column])

        plt.xlabel(columns[0])
        plt.ylabel(column)

        file_name = column.replace(" ", "_").replace("/", "_").replace(":", "_") + ".png"
        
        plt.savefig(download_dir + file_name)
        plt.close()

if __name__ == '__main__':
    try:
        os.mkdir(download_dir)
    except:
        pass

    args = sys.argv[1:]
    main(args[0])
import csv
import os
import matplotlib.pyplot as plt
from collections import Counter
from preprocessing import WORKING_PATH

INPUT = os.path.join(WORKING_PATH, 'cellnumber_summary.csv')
OUTPUT = WORKING_PATH


def generate_distribution(data, filename):
    """
    Generate histogram with matplotlib
    
    :param data: the data to be grouped
    :param filename: output destination
    """
    plt.hist(data, bins=50, facecolor='green')
    plt.xlabel('Number of masks')
    plt.ylabel('Count')
    plt.title("Distribution of masks' numbers of images in training set")
    plt.savefig(os.path.join(OUTPUT, filename))
    # plt.show()


def generate_statistics(data, filename):
    """
    Output the statistics to files
    
    :param data: the data to be output
    :param filename: output destination
    """
    with open(os.path.join(OUTPUT, filename), 'w') as f:
        writer = csv.writer(f)
        for entry in data:
            writer.writerow([entry[0], entry[1]])


if __name__ == "__main__":
    # Read from cellnumber_summary.csv file into Counter
    with open(INPUT, 'r') as f:
        reader = csv.reader(f)
        counter = Counter({rows[0]:int(rows[1]) for rows in reader})

    generate_distribution(counter.values(), 'cellnumber_distribution')

    # Find the images with most cells and least cells
    # to look for outliers
    counter_least = counter.most_common()[-1:-11:-1]
    generate_statistics(counter_least, 'top10_least_cells')

    counter_most = counter.most_common(10)
    generate_statistics(counter_most, 'top10_most_cells')

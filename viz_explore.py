import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import wrangle

def histograms1(df):
    '''
    creates some histograms to visually review the data--7 at a time
    '''
    plt.figure(figsize=(16, 3))
    # List of columns
    cols = ['bathrooms', 'bedrooms', 'sq_ft', 'full_baths','lot_size','rooms', 
            'structure_value']
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        # Create subplot.
        plt.subplot(1,7, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout(),
    plt.figure(figsize=(16, 3))
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        # Create subplot.
        plt.subplot(1,7, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        plt.boxplot(df[col])
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout()


def hists2(df):
    '''
    Same as hists1 but for the next 7 attributes
    '''
    plt.figure(figsize=(16, 3))

    # List of columns

    cols = ['tax_value', 'year_assessed', 'land_value',
        'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom']
    # col list from previous analysis...
    # cols = ['bedrooms', 'bathrooms','sq_ft','tax_value', 'age', 'sq_ft_per_bathroom']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,7, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout(),

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,7, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.boxplot(df[col])

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout()


def hists3(df):
    '''
    Same as hists2 but for the next 6 attributes
    '''
    plt.figure(figsize=(16, 3))

    # List of columns

    cols = ['sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath',
        'Los_Angeles', 'Orange', 'Ventura']
    # col list from previous analysis...
    # cols = ['bedrooms', 'bathrooms','sq_ft','tax_value', 'age', 'sq_ft_per_bathroom']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,7, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout(),

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,7, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.boxplot(df[col])

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout()



def hists4(df):
    '''
    Same as hists2 but for the final attributes
    '''
    plt.figure(figsize=(16, 3))

    # List of columns

    cols = ['tax_rate','price_per_sq_ft']
    # col list from previous analysis...
    # cols = ['bedrooms', 'bathrooms','sq_ft','tax_value', 'age', 'sq_ft_per_bathroom']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,2, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout(),

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1,2, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.boxplot(df[col])

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout()
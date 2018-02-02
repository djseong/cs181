#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math 

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:
        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

def graph(years, Y, xlabel, ylabel, xstart, xend): 
    # Create the simplest basis, with just the time and an offset.
    X = np.vstack((np.ones(years.shape), years)).T
    # a 
    X_a = np.ones(years.shape)
    for i in range(1, 6): 
        X_a = np.vstack((X_a, years ** i))
    X_a = X_a.T

    # b 
    X_b = np.ones(years.shape) 
    for i in range(1960, 2010, 5): 
        X_b = np.vstack((X_b, np.exp(-(years - i)**2/25)))
    X_b = X_b.T

    # c
    X_c = np.ones(years.shape) 
    for i in range(1, 6): 
        X_c = np.vstack((X_c, np.cos(years/i)))
    X_c = X_c.T

    # d 
    X_d = np.ones(years.shape) 
    for i in range(1, 26): 
        X_d = np.vstack((X_d, np.cos(years/i)))
    X_d = X_d.T

    # simple linear 
    # Find the regression weights using the Moore-Penrose pseudoinverse.
    w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

    # Compute the regression line on a grid of inputs.
    # DO NOT CHANGE grid_years!!!!!
    grid_years = np.linspace(xstart, xend, 200)
    grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
    grid_Yhat  = np.dot(grid_X.T, w)
    plt.plot(years, Y, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    # plotting a 
    grid_years = np.linspace(xstart, xend, 200)
    g_a = np.ones(grid_years.shape)
    for i in range(1, 6): 
        g_a = np.vstack((g_a, grid_years ** i))
    w_a = np.linalg.solve(np.dot(X_a.T, X_a) , np.dot(X_a.T, Y))
    grid_Yhat  = np.dot(g_a.T, w_a)

    plt.plot(years, Y, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    # plotting b
    if xend == 2005: 
        grid_years = np.linspace(xstart, xend, 200)
        g_b = np.ones(grid_years.shape)
        for i in range(1960, 2010, 5): 
            g_b = np.vstack((g_b, np.exp(-(grid_years - i)**2/25)))
        w_b = np.linalg.solve(np.dot(X_b.T, X_b) , np.dot(X_b.T, Y))
        grid_Yhat  = np.dot(g_b.T, w_b)

        plt.plot(years, Y, 'o', grid_years, grid_Yhat, '-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # plotting c 
    grid_years = np.linspace(xstart, xend, 200)
    g_c = np.ones(grid_years.shape)
    for i in range(1, 6): 
        g_c = np.vstack((g_c, np.cos(grid_years/i)))
    w_c = np.linalg.solve(np.dot(X_c.T, X_c) , np.dot(X_c.T, Y))
    grid_Yhat  = np.dot(g_c.T, w_c)

    plt.plot(years, Y, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    # plotting d 
    grid_years = np.linspace(xstart, xend, 200)
    g_d = np.ones(grid_years.shape)
    for i in range(1, 26): 
        g_d = np.vstack((g_d, np.cos(grid_years/i)))
    w_d = np.linalg.solve(np.dot(X_d.T, X_d) , np.dot(X_d.T, Y))
    grid_Yhat  = np.dot(g_d.T, w_d)

    plt.plot(years, Y, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


    # Nothing fancy for outputs.

    # TODO: plot and report sum of squared error for each basis
    # simple 
    pts_smp = np.dot(X, w)
    e_smp = ((Y - pts_smp)**2).sum()
    print(e_smp)

    # a 
    pts_a = np.dot(X_a, w_a)
    e_a = ((Y - pts_a)**2).sum()
    print(e_a)

    # b
    if xend == 2005:  
        pts_b = np.dot(X_b, w_b)
        e_b = ((Y - pts_b)**2).sum()
        print(e_b)

    #c 
    pts_c = np.dot(X_c, w_c)
    e_c = ((Y - pts_c)**2).sum()
    print(e_c)

    # d
    pts_d = np.dot(X_d, w_d)
    e_d = ((Y - pts_d)**2).sum()
    print(e_d)


Y = republican_counts
print("Years vs Republicans error: ")
graph(years, Y, "Years", "Number of Republicans in Congress", 1960, 2005) 
Y = republican_counts[years<last_year]
X = sunspot_counts[years<last_year]
print("Sunspots vs Republicans error: ")
graph(X, Y, "Number of Sunspots", "Number of Republicans in Congress", 10, 160) 


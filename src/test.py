import matplotlib.pyplot as plt

data = [0, 0, 1, 1] # x axis is bin size, histogram height is count of each bin, small value of y axis, small bin size?

plt.hist(data, bins=5, edgecolor='black')  # Adjust bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')

plt.show()
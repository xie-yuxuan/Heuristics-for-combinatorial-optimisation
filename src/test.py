import matplotlib.pyplot as plt

data = [0, 1, 0, 2, 0, 4, 1, 4]

plt.hist(data, bins=5, edgecolor='black')  # Adjust bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')

plt.show()
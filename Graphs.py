import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";") # reading data file and separating data by delimiter specified (;)


# summing up alcohol content column by quality of wine
sums = data.groupby(data["quality"])["alcohol"].sum()
plt.axis('equal')
per = (100. * sums/sums.sum()).__round__(1) # creating % ratio for pie chart output
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(sums.index, per)] # pie chart labels etc
plt.pie(sums,labels=sums.index) # plotting pie graph
plt.legend(labels,loc="best")
plt.title("Quality of Wine in relation to Amount of alcohol content")

plt.show() # showing pie chart
"""
hist(x=sums.index , density=1, bins=30)
ylabel('Probability');
show()"""
# getting average of sulphates content column by quality of wine
sums = data.groupby(data["quality"])["sulphates"].mean()
plt.scatter(data["sulphates"], data["quality"])# plotting scatter plot
plt.title("Quality of wine in relation to Sulphates content")
plt.xlabel("Sulphates content")
plt.ylabel("Quality")
plt.show()  # showing scatter plot
x = sums
y = sums.index
z = np.polyfit(x, y, 3) # creating trendline
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.scatter(sums, sums.index)
plt.title("Quality of wine in relation to Sulphates content mean")
plt.xlabel("Sulphates content mean")
plt.ylabel("quality")
plt.show() # showing scatter plot

sums = data.groupby(data["quality"])["alcohol"].mean()# getting average of alcohol content column by quality of wine
plt.scatter(data["alcohol"], data["quality"])# plotting scatter plot
plt.title("Quality of wine in relation to Alcohol content")
plt.xlabel("Alcohol content")
plt.ylabel("Quality")
plt.show() # showing scatter plot
plt.scatter(sums, sums.index)
x = sums
y = sums.index
z = np.polyfit(x, y, 1) # creating trendline
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.title("Quality of wine in relation to Alcohol content mean")
plt.xlabel("Alcohol contents mean")
plt.ylabel("Quality")
plt.show() # showing scatter plot


sums = data.groupby(data["quality"])["citric acid"].mean()# getting average of citric acid content column by quality of wine
plt.scatter(data["citric acid"], data["quality"])# plotting scatter plot
plt.title("Quality of wine in relation to citric acid content")
plt.xlabel("citric acid content")
plt.ylabel("Quality")
plt.show() # showing scatter plot
plt.scatter(sums, sums.index)
x = sums
y = sums.index
z = np.polyfit(x, y, 1) # creating trendline
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.title("Quality of wine in relation to citric acid content mean")
plt.xlabel("citric acid mean")
plt.ylabel("Quality")
plt.show() # showing scatter plot
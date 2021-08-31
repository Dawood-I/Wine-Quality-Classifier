import pandas as pd
import matplotlib.pyplot as plt

sal_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")

# Create training/testing datasets
from sklearn.model_selection import train_test_split# splitting data to train and test | data = 80% | testing set = 20%
train, test = train_test_split(sal_data, test_size=0.2, random_state=42)
train_data = train.drop(['citric acid','quality'], axis=1)#test/train data before scaler = all columns - quality,citric acid
train_labels = train.iloc[:,-1] # test/train label = quality column

test_labels = test.iloc[:,-1]
test_data = test.drop(["citric acid","quality"], axis=1)


# generating PCA for fitting and reducing to 2 columns
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pcatransformed = pca.fit_transform(train_data)

print(pca.explained_variance_ratio_)  #printing variance ratio


# converting pca to dataframe
pca_Df = pd.DataFrame(data = pcatransformed
             , columns = ['principal component 1', 'principal component 2'])
print(len(pca_Df))
print(len(train_labels))
# reseting index for train labels
train_labels = train_labels.reset_index(drop=True)
finalDf = pd.concat([pca_Df, train_labels], axis = 1) # concatnating the pca fitted components dat with original train labels data
finalDf = pd.DataFrame(finalDf)
colordf = finalDf # data frame where quality values are replaced with letters ment for use in depicting qualities based on color during plotting
colordf.replace({3:"r",4:"g",5:"b",6:"c",7:"y",8:"fuchsia"},inplace=True)
plt.figure(figsize=(10, 8))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.scatter(finalDf["principal component 1"],finalDf["principal component 2"], color=colordf["quality"], label='OK', marker='*') # plotting pca graph
from matplotlib.lines import Line2D
# making customized legen
custom_legend = [Line2D([0], [0], marker="*", color="r", lw=0),
                 Line2D([0], [0], marker="*", color="g", lw=0),
                 Line2D([0], [0], marker="*", color="b", lw=0),
                 Line2D([0], [0], marker="*", color="c", lw=0),
                 Line2D([0], [0], marker="*", color="y", lw=0),
                 Line2D([0], [0], marker="*", color="fuchsia", lw=0),

                 ]
plt.legend(custom_legend, [3,4,5,6,7,8], title="Quality of wine")
plt.title("Quality of wine PCA graph")
plt.grid()
plt.show()


from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

#Part 1

#print(df_wine.describe())
#print("number of rows = ", df_wine.shape[0])
#print("number of cols = ", df_wine.shape[1])
#cormat = df_wine.corr()
#print(cormat)
#plt.figure()
#hm= pd.DataFrame(df_wine.corr())
#plt.pcolor(hm)
#plt.title("Correlation Matrix")
#plt.xlabel("features")
#plt.ylabel("features")
#plt.show()
#
#plt.figure()
#cols = df_wine.columns
#sns.set(font_scale=1.5)
#cm = np.corrcoef(df_wine[cols].values.T)
#hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 5},yticklabels=cols,xticklabels=cols)
#plt.xlabel("features")
#plt.ylabel("features")
#plt.show()

#Part 2
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_y_train_pred = lr.predict(X_train)
print( "Logistic Regression train accurancy score: ",metrics.accuracy_score(y_train, lr_y_train_pred) )
lr_y_pred = lr.predict(X_test)
print( "Logistic Regression test accurancy score: ",metrics.accuracy_score(y_test, lr_y_pred) )



from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train, y_train)
svm_y_train_pred = svm.predict(X_train)
print( "SVM Regression train accurancy score: ",metrics.accuracy_score(y_train, svm_y_train_pred) )
svm_y_pred = svm.predict(X_test)
print("SVM Regression test accurancy score: ",metrics.accuracy_score(y_test, svm_y_pred) )

#Part 3
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#Logistic Regression fitted on PCA transformed dataset
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
pca_lr_y_train_pred = lr.predict(X_train_pca)
pca_lr_y_pred = lr.predict(X_test_pca)
print( "Logistic Regression(PCA) train accurancy score: ",metrics.accuracy_score(y_train, pca_lr_y_train_pred) )
print("Logistic Regression(PCA) test accurancy score: ",metrics.accuracy_score(y_test, pca_lr_y_pred) )




from matplotlib.colors import ListedColormap
#def plot_decision_regions(X, y, classifier, resolution=0.02):
#    markers = ('s', 'x', 'o', '^', 'v')
#    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#    cmap = ListedColormap(colors[:len(np.unique(y))])
#    # plot the decision surface
#    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
#    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#    plt.xlim(xx1.min(), xx1.max())
#    plt.ylim(xx2.min(), xx2.max())
#    # plot class samples
#    for idx, cl in enumerate(np.unique(y)):
#        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)
#
#
#plot_decision_regions(X_train_pca, y_train, classifier=lr)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#plt.show()








#SVM Regression fitted on PCA transformed dataset
pca = PCA(n_components=2)
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_pca, y_train)
pca_svm_y_train_pred = svm.predict(X_train_pca)
pca_svm_y_pred = svm.predict(X_test_pca)
print( "SVM Regression(PCA) train accurancy score: ",metrics.accuracy_score(y_train, pca_svm_y_train_pred) )
print("SVM Regression(PCA) test accurancy score: ",metrics.accuracy_score(y_test, pca_svm_y_pred) )

#Part 4
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)
#Logistic Regression fitted on LDA transformed dataset
lda_lr_y_train_pred = lr.predict(X_train_lda)
lda_lr_y_pred = lr.predict(X_test_lda)
print( "Logistic Regression(LDA) train accurancy score: ",metrics.accuracy_score(y_train, lda_lr_y_train_pred) )
print("Logistic Regression(LDA) test accurancy score: ",metrics.accuracy_score(y_test, lda_lr_y_pred) )
#SVM Regression fitted on LDA transformed dataset
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_lda, y_train)
lda_svm_y_train_pred = svm.predict(X_train_lda)
lda_svm_y_pred = svm.predict(X_test_lda)
print( "SVM Regression(LDA) train accurancy score: ",metrics.accuracy_score(y_train, lda_svm_y_train_pred) )
print("SVM Regression(LDA) test accurancy score: ",metrics.accuracy_score(y_test, lda_svm_y_pred) )


#Part 5
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=0.1)
X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
X_test_skernpca= scikit_kpca.transform(X_test_std)

#Logistic Regression fitted on KPCA transformed dataset
lr = LogisticRegression()
lr.fit(X_train_skernpca, y_train)
kpca_lr_y_train_pred = lr.predict(X_train_skernpca)
kpca_lr_y_pred = lr.predict(X_test_skernpca)
print( "Logistic Regression(KPCA) train accurancy score(gamma=0.1): ",metrics.accuracy_score(y_train, kpca_lr_y_train_pred) )
print("Logistic Regression(KPCA) test accurancy score(gamma=0.1): ",metrics.accuracy_score(y_test, kpca_lr_y_pred) )
#SVM Regression fitted on KPCA transformed dataset
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_skernpca, y_train)
kpca_svm_y_train_pred = svm.predict(X_train_skernpca)
kpca_svm_y_pred = svm.predict(X_test_skernpca)
print( "SVM Regression(KPCA) train accurancy score(gamma=0.1): ",metrics.accuracy_score(y_train, kpca_svm_y_train_pred) )
print("SVM Regression(KPCA) test accurancy score(gamma=0.1): ",metrics.accuracy_score(y_test, kpca_svm_y_pred) )

lr_train_accu_scores = []
lr_test_accu_scores = []
svm_train_accu_scores = []
svm_test_accu_scores = []


gamma_space = np.linspace(0.01,0.5, endpoint = True)
for gamma in gamma_space:
    scikit_kpca.gamma = gamma
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
    X_test_skernpca= scikit_kpca.transform(X_test_std)
    
    #Logistic Regression fitted on KPCA transformed dataset
    lr = LogisticRegression()
    lr.fit(X_train_skernpca, y_train)
    kpca_lr_y_train_pred = lr.predict(X_train_skernpca)
    kpca_lr_y_pred = lr.predict(X_test_skernpca)
    lr_train_accu_scores.append(metrics.accuracy_score(y_train, kpca_lr_y_train_pred))
    lr_test_accu_scores.append(metrics.accuracy_score(y_test, kpca_lr_y_pred))

    #SVM Regression fitted on KPCA transformed dataset
    svm = SVC(kernel = 'rbf', C= 1.0, random_state= 1)
    svm.fit(X_train_skernpca, y_train)
    kpca_svm_y_train_pred = svm.predict(X_train_skernpca)
    kpca_svm_y_pred = svm.predict(X_test_skernpca)
    svm_train_accu_scores.append(metrics.accuracy_score(y_train, kpca_svm_y_train_pred))
    svm_test_accu_scores.append(metrics.accuracy_score(y_test, kpca_svm_y_pred))


plt.figure()
plt.plot(gamma_space, lr_train_accu_scores, label='Logistic Regression Train Set')
plt.plot(gamma_space, lr_test_accu_scores, label='Logistic Regression Test Set')
plt.legend(loc = 1)
plt.xlabel("Gamma Values")
plt.ylabel("Accuracy Scores")
plt.title("Logistic Regression Fitted on KPCA Transformed Dataset")

plt.figure()
plt.plot(gamma_space, svm_train_accu_scores, label='SVM Regression Train Set')
plt.plot(gamma_space, svm_test_accu_scores, label='SVM Regression Test Set')
plt.legend(loc = 1)
plt.xlabel("Gamma Values")
plt.ylabel("Accuracy Scores")
plt.title("SVM Regression Fitted on KPCA Transformed Dataset")


print("My name is {Yue Liu}")
print("My NetID is: {yueliu6}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



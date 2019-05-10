import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#root = "C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/labels.npy'
#dest = root +
# Load data from numpy file
X =  np.load("C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/features1.npy")
t =  np.load("C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/labels.npy")
#print(len(t))
ac=[0,0,0,0,0,0,0,0]
for i in range(0,50):
   k = []
   for p in t:
      k.append(p[i])
   y=np.array(k)


   #print(X,y)
   # Split data into training and test subsets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   models = [BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier(),
             KNeighborsClassifier(), GaussianNB(), tree.DecisionTreeClassifier(),
             svm.SVC(C=20.0, gamma=0.00001), OutputCodeClassifier(BaggingClassifier())]
   model_names = ["Bagging with DT", "Random Forest", "AdaBoost", "KNN", "Naive Bayes", "Decision Tree",
                  "Linear SVM", "OutputCodeClassifier with Linear SVM",]
   #ac=[0,0,0,0,0,0,0,0]
   count=0
   for model, name in zip(models, model_names):
      model.fit(X_train, y_train)
   # Simple SVM
   #print('fitting...')
      prediction = model.predict(X_test)
   # Print Accuracy
      acc = accuracy_score(y_test, prediction)
   #clf = SVC(C=20.0, gamma=0.00001)
   #clf.fit(X_train, y_train)
   #acc = clf.score(X_test, y_test)
      ac[count]=ac[count]+acc
      count=count+1
for i in range(0,8):
   ac[i]=ac[i]/50
print("acc=",ac)

# models = [BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier(),
#           KNeighborsClassifier(), GaussianNB(), tree.DecisionTreeClassifier(),
#           svm.SVC(kernel='linear', C=1), OutputCodeClassifier(BaggingClassifier()),
#           OneVsRestClassifier(svm.SVC(kernel='linear'))]
#
# model_names = ["Bagging with DT", "Random Forest", "AdaBoost", "KNN", "Naive Bayes", "Decision Tree",
#                "Linear SVM", "OutputCodeClassifier with Linear SVM", "OneVsRestClassifier with Linear SVM"]
# # ----------------------------------------------------------------
# # Run Each Model
# # ----------------------------------------------------------------
# for model, name in zip(models, model_names):
#     model.fit(data_train, label_train)
#     # Display the relative importance of each attribute
#     if name == "Random Forest":
#         print(model.feature_importances_)
#         # Predict
#     prediction = model.predict(data_test)
#     # Print Accuracy
#     acc = accuracy_score(label_test, prediction)
#     print("Accuracy Using", name, ": " + str(acc) + '\n')
#     print(classification_report(label_test, prediction))
#     print(confusion_matrix(label_test, prediction))

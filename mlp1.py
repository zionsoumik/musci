import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding

from keras.layers.convolutional import Convolution1D
from keras import backend as K


#root = "C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/labels.npy'
#dest = root +
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 10
max_features = 20000
batch_size = 32
nb_classes = 2
# Load data from numpy file
X =  np.load("C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/features1.npy")
t =  np.load("C:/Users/deyso/PycharmProjects/sound/mp3folder/npy_files_TOTAL_train/labels.npy")
#print(len(t))
ac=0
for i in range(17,19):
   k = []
   for p in t:
      k.append(p[i])
   y=np.array(k)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   Y_train = np_utils.to_categorical(y_train, nb_classes)
   Y_test = np_utils.to_categorical(y_test, nb_classes)
   #print(X,y)
   # Split data into training and test subsets
   input_dim = X_train.shape[1]
   # pre-processing: divide by max and substract mean
   model = Sequential()
   # model.add(Dense(256, input_dim=input_dim))
   # model.add(Activation('relu'))
   # model.add(Dropout(0.4))
   # model.add(Dense(128))
   # model.add(Activation('relu'))
   # model.add(Dropout(0.2))
   # model.add(Dense(nb_classes))
   # model.add(Activation('softmax'))
   #
   # # we'll use categorical xent for the loss, and RMSprop as the optimizer
   # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', show_accuracy=True)
   #
   # print("Training...")
   # model.fit(X_train, Y_train, nb_epoch=5, batch_size=16, validation_split=0.1)
   #
   # print("Generating test predictions...")
   # preds = model.predict_classes(X_test, verbose=0)
   # print('prediction 6 accuracy: ', accuracy_score(test['HS'], preds))
   model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1)))
   model.add(Conv2D(32, kernel_size=3, activation=’relu’))
   model.add(Flatten())
   model.add(Dense(10, activation=’softmax’))

   model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
   print('Train...')
   model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
             validation_data=(X_test, Y_test))
   score, acc = model.evaluate(X_test, Y_test,
                               batch_size=batch_size)
   #print('Test score:', score)
   #print('Test accuracy:', acc)
   ac=ac+acc

   #print("Generating test predictions...")
   #preds = model.predict_classes(X_test, verbose=0)
   #print('prediction 8 accuracy: ', accuracy_score(test['HS'], preds))
   #clf = SVC(C=20.0, gamma=0.00001)
   #clf.fit(X_train, y_train)
   K.clear_session()
   #acc = clf.score(X_test, y_test)

print("acc=",ac/50)

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

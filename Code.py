
#Các thư viện và data vào python

import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

from sklearn.model_selection import  train_test_split

#Đẩy dataset lên

digits = datasets.load_digits()

print(digits.DESCR)

#In hình ảnh lên

plt.gray()

plt.matshow(digits.images[0])

plt.show()

#Chuẩn bị dữ liệu cho mô hình học máy

# n là mẫu nằm trong dataset

n = len(digits.images)

data = digits.images.reshape((n, -1))

#Chia train/test

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=10, shuffle=True)

#Bộ phân lớp

classifier = svm.SVC()

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

print(metrics.classification_report(y_test, predicted))

metrics.plot_confusion_matrix(classifier, X_test, y_test)

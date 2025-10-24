from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
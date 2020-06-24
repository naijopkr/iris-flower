import pandas as pd
import seaborn as sns

iris_df = sns.load_dataset('iris')

# Train test split
from sklearn.model_selection import train_test_split as tts

X = iris_df.drop('species', axis=1)
y = iris_df['species']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=101)

# Train model
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gridsearch
from sklearn.model_selection import GridSearchCV

param_grid = dict(
    C = [0.1, 1, 10, 100, 1000],
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
)

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_

y_grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test, y_grid_pred))
print(classification_report(y_test, y_grid_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

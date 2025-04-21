python -m venv ml_env
source ml_env/bin/activate  # For Windows: ml_env\Scripts\activate

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = sns.load_dataset('titanic').dropna()

data['sex'] = data['sex'].map({'male': 0, 'female': 1})
X = data[['pclass', 'sex', 'age', 'fare']]
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

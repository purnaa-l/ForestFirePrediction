
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")
data=pd.read_csv('Forest_fire.csv')
data=np.array(data)

X=data[1:, 1:-1]
y=data[1:, -1]
y=y.astype(float)
X=X.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg=LogisticRegression()
log_reg.fit(X_train, y_train)
b=log_reg.predict(X_test)

pickle.dump(log_reg, open('forest_fire_model.pkl', 'wb'))
loaded_model = pickle.load(open('forest_fire_model.pkl', 'rb'))
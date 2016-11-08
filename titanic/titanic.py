
import pandas as pd
import tensorflow as tf

COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]


CATEGORICAL_COLUMNS = ["Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare"]
LABEL_COLUMN = "Survived"

df_train = pd.read_csv("train.csv")
#df_test = pd.read_csv("test.csv")

df_train["Cabin"] = df_train["Cabin"].fillna('')
df_train["Embarked"] = df_train["Embarked"].fillna('')
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())

del df_train["PassengerId"]  # otherwise alg will memorize survival for each passenger ID
print(df_train.isnull().any())
print(df_train.head(5))
print(df_train.describe())


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

print(train_input_fn())






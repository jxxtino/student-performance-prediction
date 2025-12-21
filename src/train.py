# %% SEMMA (SAMPLE)

import pandas as pd
from sklearn import model_selection

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df = pd.read_csv("../data/raw/StudentsPerformance.csv")

target = "average score"
score_cols = ["math score", "reading score", "writing score"]

df["average score"] = df[score_cols].mean(axis=1).round(2)

features = [col for col in df.columns if col != target]

class_counts = df[target].value_counts()
valid_classes = class_counts[class_counts > 1].index

df_sampled = df[df[target].isin(valid_classes)].copy()

X, y = df_sampled[features], df_sampled[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=43,
    stratify=y
)

df_sampled.to_csv(
    "../data/interim/StudentsPerformance_Sampled", 
    index=False
    )


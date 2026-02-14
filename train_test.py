import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("gesture_landmarks_real_final.csv")

train, test = train_test_split(
    data,
    test_size=0.15,
    stratify=data["label"],
    random_state=42
)

train.to_csv("gesture_train.csv", index=False)
test.to_csv("gesture_test.csv", index=False)

print("Train:", len(train))
print("Test:", len(test))

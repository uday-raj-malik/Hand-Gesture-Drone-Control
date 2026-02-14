import pandas as pd

old = pd.read_csv("gesture_landmarks_real_final.csv")   # your 2300-row real dataset
new = pd.read_csv("up_s5.csv")

combined = pd.concat([old, new], ignore_index=True)

combined.to_csv("gesture_landmarks_real_final.csv", index=False)

print("Original:", len(old))
print("New:", len(new))
print("Final:", len(combined))

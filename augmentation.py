import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("gesture_landmarks_new.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

augmented_X = []
augmented_y = []

def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    rotated = []
    for i in range(0, len(points), 3):
        xy = np.dot(R, points[i:i+2])
        rotated.extend([xy[0], xy[1], points[i+2]])
    return np.array(rotated)

for i in range(len(X)):
    original = X[i]

    # Always keep original
    augmented_X.append(original)
    augmented_y.append(y[i])

    # ---- 1. Add Noise ----
    noise = np.random.normal(0, 0.01, original.shape)
    augmented_X.append(original + noise)
    augmented_y.append(y[i])

    # ---- 2. Scale ----
    scale = np.random.uniform(0.9, 1.1)
    augmented_X.append(original * scale)
    augmented_y.append(y[i])

    # ---- 3. Rotate small angle ----
    angle = np.random.uniform(-0.15, 0.15)
    augmented_X.append(rotate(original, angle))
    augmented_y.append(y[i])

    # ---- 4. Small shift ----
    shift = np.random.normal(0, 0.02, original.shape)
    augmented_X.append(original + shift)
    augmented_y.append(y[i])

# Convert back to dataframe
aug_df = pd.DataFrame(augmented_X, columns=df.columns[:-1])
aug_df["label"] = augmented_y

# Save new dataset
aug_df.to_csv("gesture_landmarks_augmented.csv", index=False)

print("Augmented dataset saved!")
print("Old size:", len(df))
print("New size:", len(aug_df))

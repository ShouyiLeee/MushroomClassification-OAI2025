import os

import enter
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC


def process_file(file, issort=False):
    df = pd.read_csv(file)

    X = []
    y = []
    images = []
    hash_ids = set(df['image'].values)
    if issort:
        hash_ids = sorted(hash_ids)
        print(f"Process file {file} follow sorted image.")
    class_names = df.columns[-4:].tolist()
    columns = df.columns.tolist()
    for hid in hash_ids:
        rows = df.loc[df['image'] == hid]
        images.append(hid)
        if 'actual' in columns:
            y.append(class_names.index(rows['actual'].tolist()[0]))

        feat = []
        for label in class_names:
            att = rows[label].tolist()
            # Calculate mean for even indices
            att_0 = np.mean([att[i] for i in range(len(att)) if i % 2 == 0])
            att_1 = np.mean([att[i] for i in range(len(att)) if i % 2 == 1])
            att = [att_0, att_1]
            assert len(att) == 2  # !!! use 2 type is vit and cnn
            feat.extend(att)
        X.append(feat)

    X = np.array(X)
    y = np.array(y)
    return X, y, images, class_names


def train_lr(X, y, images, class_names, X_test, images_test):
    # Create individual classifiers
    linear_model = LogisticRegression()
    degree = 2
    poly_model = make_pipeline(
        PolynomialFeatures(degree), LogisticRegression())

    # Create a Voting Classifier
    voting_classifier = VotingClassifier(estimators=[
        ('linear', linear_model),
        ('polynomial', poly_model)
    ], voting='soft')  # Use 'soft' for probability-based voting

    # Fit the Voting Classifier
    voting_classifier.fit(X, y)

    # Make predictions
    # ensemble_predictions = voting_classifier.predict(X_test)
    ensemble_probs = hand_predict(X_test=X_test)
    print("[!] Use hand predict.")

    # Now you can use ensemble_predictions for further analysis or evaluation
    if len(X_test) == 200:
        # Get class with highest probability
        ensemble_predictions = [np.argmax(prob) for prob in ensemble_probs]
        print("Result:", ensemble_predictions)
        print("Score:", voting_classifier.score(X, y))
        print("Score:", voting_classifier.score(
            X_test, [0] * 50 + [3] * 50 + [1] * 50 + [2] * 50))

    order_mapping = {
        0: 1,  # position 1 in original has class 0 "nấm mỡ"
        1: 3,  # position 3 in original has class 1 "bào ngư xám+trắng"
        2: 0,  # position 0 in original has class 2 "Đùi gà baby"
        3: 2   # position 2 in original has class 3 "linh chi trắng"
    }

    df_dict = {'image_name': images_test}

    # Add probability columns
    for i, class_name in enumerate(class_names):
        original_pos = order_mapping[i]
        print(class_names[original_pos])
        prob_column = f"prob_{original_pos}"
        df_dict[prob_column] = [probs[i] for probs in ensemble_probs]

    df = pd.DataFrame(df_dict)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(
        os.getcwd(), enter.FINAL_TO_USE_DIR), exist_ok=True)
    output_file = os.path.join(
        os.getcwd(), enter.FINAL_TO_USE_DIR, "result_XceptionVIT.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


def hand_predict(X_test):
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum(axis=0)

    predictions = []
    for i in range(len(X_test)):  # X[i] is size (8)
        x0 = [X_test[i][j] for j in range(len(X_test[i])) if j % 2 == 0]
        x1 = [X_test[i][j] for j in range(len(X_test[i])) if j % 2 == 1]
        x = (softmax(x0) + softmax(x1)) / 2
        predictions.append(x)
        # print(f"Image {i}: {x}")

    return predictions


if __name__ == "__main__":
    X, y, images, class_names = process_file("valid.csv")  # Fixed
    tmp = process_file("test.csv", issort=True)  # Fixed
    train_lr(X, y, images, class_names, tmp[0], tmp[2])

import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import split_data

# Threshold for acceptable accuracy
ACCURACY_THRESHOLD = 0.9

@pytest.fixture
def load_data():
    """Fixture to load and prepare the dataset."""
    df = pd.read_csv("./data/Iris.csv")
    return split_data(df)

def test_model_validation(load_data):
    """
    Test the validation of a DecisionTreeClassifier model.
    Ensures that the accuracy meets the defined threshold.
    """
    # Unpack the data
    X_train, X_test, y_train, y_test = load_data

    # Step 1: Initialize the model
    clf = DecisionTreeClassifier(criterion="gini")

    # Step 2: Train the model
    clf.fit(X_train, y_train)

    # Step 3: Make predictions
    y_pred = clf.predict(X_test)

    # Step 4: Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Assert that the model accuracy is above the threshold
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Model accuracy is below the acceptable threshold. "
        f"Expected at least {ACCURACY_THRESHOLD * 100:.2f}%, but got {accuracy * 100:.2f}%."
    )

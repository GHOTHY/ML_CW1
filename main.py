import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Data Preprocessing
le = LabelEncoder()

# Function to preprocess columns
def preprocess_column(column):
    try:
        if column.dtype == 'object':
            return le.fit_transform(column)
        return column.astype(float)
    except Exception as e:
        print(f"Error processing column: {e}")
        return None

# Apply preprocessing to categorical columns
for col in data.columns:
    data[col] = preprocess_column(data[col])

# One-hot encoding for remaining categorical variables
data = pd.get_dummies(data, columns=['Developer', 'Publisher'])

# Drop rows where preprocessing failed
data = data.dropna()

# Define features (X) and target variable (y) for each task
X_genre = data.drop('Genre', axis=1)
y_genre = data['Genre']

X_platform = data.drop('Platform', axis=1)
y_platform = data['Platform']

X_rating = data.drop('Rating', axis=1)
y_rating = data['Rating']

X_critic_score = data.drop('Critic_Score', axis=1)
y_critic_score = data['Critic_Score']

# Split the dataset into training and testing sets
X_train_genre, X_test_genre, y_train_genre, y_test_genre = train_test_split(X_genre, y_genre, test_size=0.2,
                                                                            random_state=42)
X_train_platform, X_test_platform, y_train_platform, y_test_platform = train_test_split(X_platform, y_platform,
                                                                                        test_size=0.2, random_state=42)
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(X_rating, y_rating, test_size=0.2,
                                                                                random_state=42)
X_train_critic_score, X_test_critic_score, y_train_critic_score, y_test_critic_score = train_test_split(X_critic_score,
                                                                                                        y_critic_score,
                                                                                                        test_size=0.2,
                                                                                                        random_state=42)


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # If y is a DataFrame, extract the values
    y_train = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train
    y_test = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))


# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()

# Genre Prediction
print("Genre Prediction:")
train_and_evaluate(rf_classifier, X_train_genre, y_train_genre, X_test_genre, y_test_genre)
train_and_evaluate(knn_classifier, X_train_genre, y_train_genre, X_test_genre, y_test_genre)

# Platform Prediction
print("\nPlatform Prediction:")
train_and_evaluate(rf_classifier, X_train_platform, y_train_platform, X_test_platform, y_test_platform)
train_and_evaluate(knn_classifier, X_train_platform, y_train_platform, X_test_platform, y_test_platform)

# Rating Prediction
print("\nRating Prediction:")
train_and_evaluate(rf_classifier, X_train_rating, y_train_rating, X_test_rating, y_test_rating)
train_and_evaluate(knn_classifier, X_train_rating, y_train_rating, X_test_rating, y_test_rating)

# Critic Score Classification
print("\nCritic Score Classification:")
train_and_evaluate(rf_classifier, X_train_critic_score, y_train_critic_score, X_test_critic_score, y_test_critic_score)
train_and_evaluate(knn_classifier, X_train_critic_score, y_train_critic_score, X_test_critic_score, y_test_critic_score)

def train_and_evaluate_visualize(model, X_train, y_train, X_test, y_test, class_labels):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # If y is a DataFrame, extract the values
    y_train = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train
    y_test = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test

    # Convert y_test to a NumPy array if it's a DataFrame
    y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_labels, model.__class__.__name__)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Class labels for each task
genre_labels = data['Genre'].unique()
platform_labels = data['Platform'].unique()
rating_labels = data['Rating'].unique()
critic_score_labels = data['Critic_Score'].unique()

# Genre Prediction - Random Forest Classifier
print("Genre Prediction - Random Forest Classifier:")
train_and_evaluate_visualize(rf_classifier, X_train_genre, y_train_genre, X_test_genre, y_test_genre, genre_labels)

# Genre Prediction - K-Nearest Neighbors Classifier
print("\nGenre Prediction - K-Nearest Neighbors Classifier:")
train_and_evaluate_visualize(knn_classifier, X_train_genre, y_train_genre, X_test_genre, y_test_genre, genre_labels)

# Platform Prediction - Random Forest Classifier
print("\nPlatform Prediction - Random Forest Classifier:")
train_and_evaluate_visualize(rf_classifier, X_train_platform, y_train_platform, X_test_platform, y_test_platform, platform_labels)

# Platform Prediction - K-Nearest Neighbors Classifier
print("\nPlatform Prediction - K-Nearest Neighbors Classifier:")
train_and_evaluate_visualize(knn_classifier, X_train_platform, y_train_platform, X_test_platform, y_test_platform, platform_labels)

# Rating Prediction - Random Forest Classifier
print("\nRating Prediction - Random Forest Classifier:")
train_and_evaluate_visualize(rf_classifier, X_train_rating, y_train_rating, X_test_rating, y_test_rating, rating_labels)

# Rating Prediction - K-Nearest Neighbors Classifier
print("\nRating Prediction - K-Nearest Neighbors Classifier:")
train_and_evaluate_visualize(knn_classifier, X_train_rating, y_train_rating, X_test_rating, y_test_rating, rating_labels)

# Critic Score Classification - Random Forest Classifier
print("\nCritic Score Classification - Random Forest Classifier:")
train_and_evaluate_visualize(rf_classifier, X_train_critic_score, y_train_critic_score, X_test_critic_score, y_test_critic_score, critic_score_labels)

# Critic Score Classification - K-Nearest Neighbors Classifier
print("\nCritic Score Classification - K-Nearest Neighbors Classifier:")
train_and_evaluate_visualize(knn_classifier, X_train_critic_score, y_train_critic_score, X_test_critic_score, y_test_critic_score, critic_score_labels)


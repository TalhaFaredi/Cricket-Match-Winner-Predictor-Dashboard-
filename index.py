import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

# Load dataset with new caching method
@st.cache_data
def load_data():
    data = pd.read_csv('ODI_Match_info.csv')
    return data

data = load_data()

st.title('Match winner Prediction and Model Evaluation')

# Display dataset
if st.checkbox('Show Dataset'):
    st.write(data.head())

# Select teams and venues
teams = data['team1'].unique().tolist()
venues = data['venue'].unique().tolist()

# User input for prediction
st.write('### Match winner Prediction')
team1 = st.selectbox('Select Team 1', teams)
team2 = st.selectbox('Select Team 2', teams)
venue = st.selectbox('Select venue', venues)

# Preprocessing the data
# Combine team1 and team2 into a single feature
data['team1_team2'] = data['team1'] + "_" + data['team2']

# Label Encoding the combined team and venue columns
label_encoder_team = LabelEncoder()
data['team1_team2_encoded'] = label_encoder_team.fit_transform(data['team1_team2'])
label_encoder_venue = LabelEncoder()
data['venue_encoded'] = label_encoder_venue.fit_transform(data['venue'])
label_encoder_winner = LabelEncoder()
data['winner_encoded'] = label_encoder_winner.fit_transform(data['winner'])

# Features and target label
X = data[['team1_team2_encoded', 'venue_encoded']]
y = data['winner_encoded']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection (RFE)
dt = DecisionTreeClassifier()
rfe = RFE(dt, n_features_to_select=1, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Model selection
model_choice = st.selectbox('Select Model', ['Random Forest', 'Decision Tree'])

# Train the model with hyperparameter tuning (GridSearchCV)
if model_choice == 'Decision Tree':
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train_rfe, y_train)
    best_params = grid_search.best_params_
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train_rfe, y_train)
else:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_rfe, y_train)
    best_params = grid_search.best_params_
    model = RandomForestClassifier(**best_params)
    model.fit(X_train_rfe, y_train)

y_pred = model.predict(X_test_rfe)

# Model Evaluation
st.write('### Model Evaluation')

accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
st.pyplot(fig)

# Display Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.write('### Classification Report')
st.write(pd.DataFrame(report).transpose())

# Overfitting/Underfitting Check
st.write('### Training vs Testing Accuracy')
train_acc = model.score(X_train_rfe, y_train)
test_acc = model.score(X_test_rfe, y_test)
st.write(f'Training Accuracy: {train_acc:.2f}')
st.write(f'Testing Accuracy: {test_acc:.2f}')

if train_acc > test_acc:
    st.write("The model might be **overfitting**.")
elif train_acc < test_acc:
    st.write("The model might be **underfitting**.")
else:
    st.write("The model is performing well.")

# Prediction based on user input
st.write('### Predict winner')

# Check if the selected team combination exists in the training data
input_team1_team2 = team1 + "_" + team2
if input_team1_team2 in data['team1_team2'].values:
    input_team1_team2_encoded = label_encoder_team.transform([input_team1_team2])[0]
    input_venue_encoded = label_encoder_venue.transform([venue])[0]
    input_features = [[input_team1_team2_encoded, input_venue_encoded]]
    input_features_rfe = rfe.transform(input_features)
    
    # Make prediction
    prediction_encoded = model.predict(input_features_rfe)[0]
    predicted_winner = label_encoder_winner.inverse_transform([prediction_encoded])[0]
    st.write(f"The predicted winner is: **{predicted_winner}**")
else:
    st.write(f"The combination '{team1} vs {team2}' has not been seen before in the training data.")
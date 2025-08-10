# Step 1: Importing the required libraries
import pandas as pd
import numpy as np
from collections import Counter

# Set random seed for reproducibility
starting_point = 42
np.random.seed(starting_point)
number_of_students = 500

# Generating dataset with the following features
cgpa_scores = np.round(np.random.uniform(5.0, 10.0, number_of_students), 2)
tenth_scores = np.round(np.random.uniform(50.0, 100.0, number_of_students), 2)
twelfth_scores = np.round(np.random.uniform(50.0, 100.0, number_of_students), 2)
internships_completed = np.random.randint(0, 4, number_of_students)
certifications_earned = np.random.randint(0, 6, number_of_students)
communication_skill_score = np.random.randint(1, 11, number_of_students)
technical_skill_score = np.random.randint(1, 11, number_of_students)
aptitude_test_score = np.round(np.random.uniform(0, 100, number_of_students), 2)
projects_completed = np.random.randint(0, 7, number_of_students)
hackathon_participation = np.random.randint(0, 6, number_of_students)
problem_solving_skill = np.random.randint(1, 11, number_of_students)
interview_performance_score = np.random.randint(1, 11, number_of_students)

# Creating DataFrame
student_dataset = pd.DataFrame({
    "CGPA": cgpa_scores,
    "10th_Percentage": tenth_scores,
    "12th_Percentage": twelfth_scores,
    "Internships": internships_completed,
    "Certifications": certifications_earned,
    "Communication_Skill": communication_skill_score,
    "Technical_Skill": technical_skill_score,
    "Aptitude_Score": aptitude_test_score,
    "Projects": projects_completed,
    "Hackathons": hackathon_participation,
    "Problem_Solving": problem_solving_skill,
    "Interview_Score": interview_performance_score
})

# Saving the dataset to Excel file
student_dataset.to_excel("student_dataset.xlsx", index=False)
print("Dataset has been generated and saved as student_dataset.xlsx")

# Previewing first 10 students
print(student_dataset.head(10))


# Step 2: Adding Placement Status and Salary Labels
# Creating empty lists to hold labels
placement_status = []
salary_offered = []

# Multipliers for score calculation:
# CGPA × 10 (scale to 100)
# 10th, 12th already out of 100
# Internships × 25
# Certifications × 20
# Skills × 10 (out of 10 → 100 scale)
# Aptitude already out of 100
# Projects × 20
# Hackathons × 15
for i in range(number_of_students):
    score = (
        cgpa_scores[i] * 10 +
        tenth_scores[i] +
        twelfth_scores[i] +
        internships_completed[i] * 25 +
        certifications_earned[i] * 20 +
        communication_skill_score[i] * 10 +
        technical_skill_score[i] * 10 +
        aptitude_test_score[i] +
        projects_completed[i] * 20 +
        hackathon_participation[i] * 15 +
        problem_solving_skill[i] * 10 +
        interview_performance_score[i] * 10
    )

    if score >= 800:  # Product company
        placement_status.append(2)
        salary_offered.append(np.random.randint(25000, 100001))
    elif score >= 600:  # Service company
        placement_status.append(1)
        salary_offered.append(np.random.randint(15000, 25001))
    else:  # Not placed
        placement_status.append(0)
        salary_offered.append(0)

# Adding the new columns to DataFrame
student_dataset["Placement_Status"] = placement_status
student_dataset["Salary_Offered"] = salary_offered

# Saving the updated dataset with labels to Excel again
student_dataset.to_excel("student_dataset_with_labels.xlsx", index=False)
print("Placement status and salary labels added and saved to student_dataset_with_labels.xlsx")

# Optional: Show distribution of placement status
print("Placement counts:", Counter(placement_status))


# Step 3: Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Features - exclude the target columns
X = student_dataset.drop(columns=["Placement_Status", "Salary_Offered"])

# Targets
y_placement = student_dataset["Placement_Status"]
y_salary = student_dataset["Salary_Offered"]

# Spliting data for classification (placement status)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_placement, test_size=0.2, random_state=starting_point
)

# Spliting data for regression (salary) - use only placed students (salary > 0)
placed_students = student_dataset[student_dataset["Salary_Offered"] > 0]
X_placed = placed_students.drop(columns=["Placement_Status", "Salary_Offered"])
y_salary_placed = placed_students["Salary_Offered"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_placed, y_salary_placed, test_size=0.2, random_state=starting_point
)

# Training classifier
clf = RandomForestClassifier(random_state=starting_point)
clf.fit(X_train_cls, y_train_cls)

# Predicting and evaluating the classifier
y_pred_cls = clf.predict(X_test_cls)
print("Placement Prediction Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls))

# Training regressor
reg = RandomForestRegressor(random_state=starting_point)
reg.fit(X_train_reg, y_train_reg)

# Predicting and evaluating regressor
y_pred_reg = reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Salary Prediction Mean Squared Error: {mse:.2f}")

# Saving the trained models for later use in the web app
import joblib
joblib.dump(clf, "placement_classifier.pkl")
joblib.dump(reg, "salary_regressor.pkl")

print("Models trained and saved successfully.")


# Step 4: Testing the model with new sample data
# Creating a new student profile for testing
new_student = pd.DataFrame({
    "CGPA": [8.7],
    "10th_Percentage": [88.0],
    "12th_Percentage": [85.0],
    "Internships": [2],
    "Certifications": [3],
    "Communication_Skill": [8],
    "Technical_Skill": [9],
    "Aptitude_Score": [78.0],
    "Projects": [4],
    "Hackathons": [1],
    "Problem_Solving": [8],
    "Interview_Score": [9]
})

# Predict placement
placement_prediction = clf.predict(new_student)[0]
salary_prediction = reg.predict(new_student)[0] if placement_prediction != 0 else 0

print("\n--- Test Prediction for New Student ---")
print(f"Predicted Placement Status: {placement_prediction}")
print(f"Predicted Salary: {salary_prediction:.2f}")

# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better readability
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
print("\nPreview of training data:")
print(train.head())


print("\nDataset Info:")
train.info()

print("\nSummary Statistics:")
print(train.describe())

print("\nMissing Values:")
print(train.isnull().sum())


# Fill missing 'Age' with median
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Fill missing 'Cabin' with 'Unknown'
train['Cabin'].fillna('Unknown', inplace=True)

print("\nMissing values after cleaning:")
print(train.isnull().sum())



# 5.1 Survival Count
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Survived',hue='Survived' ,palette='Set2' ,legend=False)
plt.title('Survival Count')
plt.show()

# 5.2 Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Sex', hue='Survived', palette='pastel')
plt.title('Survival by Gender')
plt.show()

# 5.3 Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Pclass', hue='Survived', palette='muted')
plt.title('Survival by Passenger Class')
plt.show()

# 5.4 Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# 5.5 Survival by Age
plt.figure(figsize=(8,5))
sns.boxplot(data=train, x='Survived', y='Age', palette='coolwarm')
plt.title('Survival vs Age')
plt.show()

# 5.6 Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()


# Average survival by class and gender
plt.figure(figsize=(8,5))
sns.barplot(data=train, x='Pclass', y='Survived', hue='Sex', palette='husl')
plt.title('Average Survival Rate by Class and Gender')
plt.show()


print("""
   Key Insights:
1. Females had a much higher survival rate than males.
2. Passengers in higher classes (1st class) had better chances of survival.
3. Younger passengers tended to survive more than older ones.
4. The 'Embarked' feature may also have some influence on survival.
5. Missing data was mostly in 'Age' and 'Cabin', which were handled appropriately.
""")
import warnings
warnings.filterwarnings("ignore")

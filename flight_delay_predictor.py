"""
Flight Delay Predictor - Machine Learning Project
Author: Your Name
Description: Predicts flight delays using machine learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("✈️  FLIGHT DELAY PREDICTOR - ML PROJECT")
print("="*60)
print("\n🔄 Step 1: Generating Sample Airline Data...")

# Generate realistic flight data
np.random.seed(42)
n_flights = 10000

# Create dataset
data = {
    'MONTH': np.random.randint(1, 13, n_flights),
    'DAY': np.random.randint(1, 32, n_flights),
    'DAY_OF_WEEK': np.random.randint(1, 8, n_flights),
    'AIRLINE': np.random.choice(['AA', 'DL', 'UA', 'WN', 'B6'], n_flights),
    'ORIGIN_AIRPORT': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], n_flights),
    'DEST_AIRPORT': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], n_flights),
    'SCHEDULED_DEPARTURE': np.random.randint(0, 2400, n_flights),
    'DISTANCE': np.random.randint(200, 3000, n_flights),
    'DEPARTURE_DELAY': np.random.randint(-20, 120, n_flights)
}

df = pd.DataFrame(data)

# Remove same origin and destination
df = df[df['ORIGIN_AIRPORT'] != df['DEST_AIRPORT']].reset_index(drop=True)

# Create target variable: 1 if delayed more than 15 minutes, 0 otherwise
df['IS_DELAYED'] = (df['DEPARTURE_DELAY'] > 15).astype(int)

# Add realistic patterns (morning flights less likely to be delayed)
morning_mask = (df['SCHEDULED_DEPARTURE'] < 1200)
df.loc[morning_mask, 'IS_DELAYED'] = (np.random.random(morning_mask.sum()) > 0.7).astype(int)

print(f"✅ Generated {len(df)} flight records")
print(f"   - Delayed flights: {df['IS_DELAYED'].sum()} ({df['IS_DELAYED'].mean()*100:.1f}%)")
print(f"   - On-time flights: {(1-df['IS_DELAYED']).sum()} ({(1-df['IS_DELAYED'].mean())*100:.1f}%)")

print("\n📊 Step 2: Data Overview")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")

# Feature Engineering
print("\n🔧 Step 3: Feature Engineering...")
df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
df['IS_MORNING'] = (df['HOUR'] < 12).astype(int)
df['IS_EVENING'] = (df['HOUR'] >= 18).astype(int)
df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 6).astype(int)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['AIRLINE_ENCODED'] = le_airline.fit_transform(df['AIRLINE'])
df['ORIGIN_ENCODED'] = le_origin.fit_transform(df['ORIGIN_AIRPORT'])
df['DEST_ENCODED'] = le_dest.fit_transform(df['DEST_AIRPORT'])

print("✅ Created new features: HOUR, IS_MORNING, IS_EVENING, IS_WEEKEND")

# Prepare features for modeling
feature_cols = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE_ENCODED', 'ORIGIN_ENCODED', 
                'DEST_ENCODED', 'HOUR', 'DISTANCE', 'IS_MORNING', 
                'IS_EVENING', 'IS_WEEKEND']

X = df[feature_cols]
y = df['IS_DELAYED']

print(f"\n🎯 Step 4: Splitting Data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# Train Model
print("\n🤖 Step 5: Training Random Forest Model...")
print("   (This may take 10-20 seconds...)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make Predictions
print("\n🔮 Step 6: Making Predictions...")
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy*100:.2f}%")
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

# Feature Importance
print("\n🔍 Step 7: Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Visualizations
print("\n📊 Step 8: Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Flight Delay Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# 2. Feature Importance
top_features = feature_importance.head(7)
axes[0,1].barh(top_features['Feature'], top_features['Importance'], color='skyblue')
axes[0,1].set_title('Top 7 Important Features')
axes[0,1].set_xlabel('Importance')

# 3. Delays by Airline
airline_delays = df.groupby('AIRLINE')['IS_DELAYED'].mean().sort_values(ascending=False)
axes[1,0].bar(airline_delays.index, airline_delays.values, color='coral')
axes[1,0].set_title('Delay Rate by Airline')
axes[1,0].set_ylabel('Delay Rate')
axes[1,0].set_xlabel('Airline')

# 4. Delays by Hour of Day
hourly_delays = df.groupby('HOUR')['IS_DELAYED'].mean()
axes[1,1].plot(hourly_delays.index, hourly_delays.values, marker='o', linewidth=2, color='green')
axes[1,1].set_title('Delay Rate by Hour of Day')
axes[1,1].set_xlabel('Hour')
axes[1,1].set_ylabel('Delay Rate')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flight_delay_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'flight_delay_analysis.png'")

# Save the model
import joblib
joblib.dump(model, 'flight_delay_model.pkl')
print("✅ Model saved as 'flight_delay_model.pkl'")

# Save encoders
joblib.dump(le_airline, 'airline_encoder.pkl')
joblib.dump(le_origin, 'origin_encoder.pkl')
joblib.dump(le_dest, 'dest_encoder.pkl')
print("✅ Encoders saved")

# Test the model with sample predictions
print("\n🧪 Step 9: Testing with Sample Predictions...")
print("\n" + "="*60)

sample_flights = [
    {'desc': 'Morning flight, short distance', 'data': [6, 2, 0, 1, 2, 8, 500, 1, 0, 0]},
    {'desc': 'Evening flight, long distance', 'data': [12, 5, 1, 0, 3, 19, 2500, 0, 1, 0]},
    {'desc': 'Weekend afternoon flight', 'data': [7, 7, 2, 1, 4, 14, 1200, 0, 0, 1]}
]

for i, flight in enumerate(sample_flights, 1):
    prediction = model.predict([flight['data']])[0]
    probability = model.predict_proba([flight['data']])[0]
    
    status = "⚠️ DELAYED" if prediction == 1 else "✅ ON-TIME"
    print(f"\nSample {i}: {flight['desc']}")
    print(f"   Prediction: {status}")
    print(f"   Confidence: {max(probability)*100:.1f}%")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Files Created:")
print("   1. flight_delay_analysis.png - Visualization dashboard")
print("   2. flight_delay_model.pkl - Trained ML model")
print("   3. airline_encoder.pkl - Airline encoder")
print("   4. origin_encoder.pkl - Origin airport encoder")
print("   5. dest_encoder.pkl - Destination airport encoder")
print("\n💡 Next Steps:")
print("   - Check the PNG file for visualizations")
print("   - Use the model for predictions in other scripts")
print("   - Add this project to your GitHub!")
print("\n✨ Great job completing your first ML project! ✨\n")

plt.show()
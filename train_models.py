import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting CyberShield ML Model Training...")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Step 1: Load the dataset
try:
    print("📊 Loading dataset...")
    df = pd.read_csv("sample_data.csv")
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display basic info about the dataset
    print(f"📋 Dataset columns: {list(df.columns)}")
    if "Label" in df.columns:
        print(f"🏷️ Label distribution:\n{df['Label'].value_counts()}")
    else:
        print("⚠️ Warning: No 'Label' column found. Creating synthetic labels...")
        # Create synthetic labels for demonstration
        np.random.seed(42)
        df['Label'] = np.random.randint(0, 3, size=len(df))
        
except FileNotFoundError:
    print("❌ Error: sample_data.csv not found!")
    print("Please ensure the dataset file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    exit(1)

# Step 2: Preprocess data
try:
    print("🔧 Preprocessing data...")
    
    # Extract numerical features (excluding Label)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in numerical_cols:
        numerical_cols.remove("Label")
    
    X = df[numerical_cols]
    y = df["Label"]
    
    print(f"✅ Features: {X.shape[1]} numerical columns")
    print(f"✅ Target: {len(y.unique())} unique classes")
    
    if X.empty:
        print("❌ Error: No numerical features found!")
        exit(1)
        
except Exception as e:
    print(f"❌ Error during preprocessing: {str(e)}")
    exit(1)

# Train-test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
except Exception as e:
    print(f"❌ Error during train-test split: {str(e)}")
    exit(1)

# Step 3: Train Decision Tree
try:
    print("\n🌳 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
    print(f"✅ Decision Tree Accuracy: {dt_accuracy:.4f}")
    
    # Save model
    pickle.dump(dt_model, open("models/dt_model.pkl", "wb"))
    print("💾 Decision Tree model saved!")
    
except Exception as e:
    print(f"❌ Error training Decision Tree: {str(e)}")

# Step 4: Train XGBoost
try:
    print("\n🚀 Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False, 
        eval_metric="mlogloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"✅ XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Save model
    pickle.dump(xgb_model, open("models/xgb_model.pkl", "wb"))
    print("💾 XGBoost model saved!")
    
except Exception as e:
    print(f"❌ Error training XGBoost: {str(e)}")

# Step 5: Train Neural Network
try:
    print("\n🧠 Training Neural Network...")
    
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Build model
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    nn_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Train with early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = nn_model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Evaluate
    nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"✅ Neural Network Accuracy: {nn_accuracy:.4f}")
    
    # Save model
    nn_model.save("models/nn_model.h5")
    print("💾 Neural Network model saved!")
    
except Exception as e:
    print(f"❌ Error training Neural Network: {str(e)}")

# Step 6: Generate detailed classification reports
try:
    print("\n📊 Generating Classification Reports...")
    
    # Decision Tree Report
    if 'dt_model' in locals():
        dt_pred = dt_model.predict(X_test)
        print("\n🌳 Decision Tree Classification Report:")
        print(classification_report(y_test, dt_pred))
    
    # XGBoost Report  
    if 'xgb_model' in locals():
        xgb_pred = xgb_model.predict(X_test)
        print("\n🚀 XGBoost Classification Report:")
        print(classification_report(y_test, xgb_pred))
    
    # Neural Network Report
    if 'nn_model' in locals():
        nn_pred = np.argmax(nn_model.predict(X_test), axis=1)
        print("\n🧠 Neural Network Classification Report:")
        print(classification_report(y_test, nn_pred))

except Exception as e:
    print(f"❌ Error generating reports: {str(e)}")

print("\n🎉 Model training completed!")
print("📁 All models saved in the 'models/' directory")
print("🚀 You can now run the Streamlit app: streamlit run app.py")

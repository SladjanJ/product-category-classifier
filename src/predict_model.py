#!/usr/bin/env python3
"""
Interactive product category predictor.
Load trained model and predict category from product title.
"""
import pandas as pd
import joblib
import sys

def compute_features(title):
    """Extract features from product title."""
    return {
        "Product Title": [title],
        "title_length": [len(title)],
        "word_count": [len(title.split())],
        "has_number": [bool(any(c.isdigit() for c in title))]
    }

def predict_category(model_path="../models/final_svm_model.pkl"):
    """Load model and predict interactively."""
    try:
        # Load model
        model = joblib.load(model_path)
        print("✅ Model loaded! Enter product titles (Ctrl+C to exit)")
        print("-" * 50)
        
        while True:
            title = input("\n📦 Product title: ").strip()
            if not title:
                continue
                
            # Extract features
            test_df = pd.DataFrame(compute_features(title))
            
            # Predict
            prediction = model.predict(test_df)[0]
            confidence = model.decision_function(test_df).max()
            
            print(f"🎯 Predicted: **{prediction}** (confidence: {confidence:.2f})")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    predict_category()

"""
Fraud Detection System for SME Financial Institutions
Author: Saiful (saifulcu105-lang)
Description: Real-time transaction fraud detection using Machine Learning
NIW Evidence: Protects US community banks and their customers from fraud
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GENERATE SAMPLE TRANSACTION DATA
# ─────────────────────────────────────────────
def generate_sample_data(n_samples=10000, fraud_rate=0.02):
    """
    Generate realistic synthetic bank transaction data for SME banks.
    In production: replace with real transaction data from your institution.
    Dataset reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    np.random.seed(42)

    # Legitimate transactions
    n_legit = int(n_samples * (1 - fraud_rate))
    n_fraud = int(n_samples * fraud_rate)

    legit = pd.DataFrame({
        'transaction_amount':   np.random.lognormal(4, 1.5, n_legit),
        'transaction_hour':     np.random.randint(8, 20, n_legit),
        'customer_age':         np.random.randint(25, 70, n_legit),
        'account_balance':      np.random.lognormal(8, 1, n_legit),
        'num_transactions_day': np.random.poisson(3, n_legit),
        'distance_from_home':   np.random.exponential(10, n_legit),
        'is_foreign':           np.random.binomial(1, 0.05, n_legit),
        'is_online':            np.random.binomial(1, 0.3, n_legit),
        'is_fraud':             0
    })

    # Fraudulent transactions (different pattern)
    fraud = pd.DataFrame({
        'transaction_amount':   np.random.lognormal(6, 2, n_fraud),
        'transaction_hour':     np.random.choice([0,1,2,3,22,23], n_fraud),
        'customer_age':         np.random.randint(18, 40, n_fraud),
        'account_balance':      np.random.lognormal(6, 2, n_fraud),
        'num_transactions_day': np.random.poisson(10, n_fraud),
        'distance_from_home':   np.random.exponential(200, n_fraud),
        'is_foreign':           np.random.binomial(1, 0.6, n_fraud),
        'is_online':            np.random.binomial(1, 0.8, n_fraud),
        'is_fraud':             1
    })

    data = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return data


# ─────────────────────────────────────────────
# 2. FRAUD DETECTION MODEL
# ─────────────────────────────────────────────
class FraudDetector:
    """
    AI-powered fraud detection for SME banks and credit unions.
    Protects community financial institutions serving rural Americans.
    """

    def __init__(self):
        self.features = [
            'transaction_amount', 'transaction_hour', 'customer_age',
            'account_balance', 'num_transactions_day',
            'distance_from_home', 'is_foreign', 'is_online'
        ]
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'   # handles fraud imbalance
            ))
        ])
        self.is_trained = False

    def train(self, data):
        """Train the fraud detection model."""
        print("\n🔄 Training Fraud Detection Model...")
        X = data[self.features]
        y = data['is_fraud']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        print("\n" + "="*55)
        print("  FRAUD DETECTION MODEL — PERFORMANCE RESULTS")
        print("="*55)
        print(f"  Precision : {precision_score(y_test, y_pred):.1%}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.1%}")
        print(f"  F1 Score  : {f1_score(y_test, y_pred):.1%}")
        print(f"  AUC-ROC   : {roc_auc_score(y_test, y_prob):.1%}")
        print("="*55)
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Legitimate', 'Fraud']))
        return self

    def predict_transaction(self, transaction: dict):
        """
        Score a single transaction in real time.
        Returns risk score and recommendation.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        df = pd.DataFrame([transaction])[self.features]
        prob = self.model.predict_proba(df)[0][1]
        score = int(prob * 100)

        if score < 30:
            risk, action, emoji = "LOW",    "✅ AUTO-APPROVE", "🟢"
        elif score < 60:
            risk, action, emoji = "MEDIUM", "⚠️  HUMAN REVIEW", "🟡"
        else:
            risk, action, emoji = "HIGH",   "🚨 BLOCK & ALERT", "🔴"

        return {
            'fraud_probability': f"{prob:.1%}",
            'risk_score':        score,
            'risk_level':        risk,
            'action':            action,
            'emoji':             emoji
        }


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTOR (Unsupervised)
# ─────────────────────────────────────────────
class AnomalyDetector:
    """
    Unsupervised anomaly detection — catches NEW fraud patterns
    that supervised models have never seen before.
    """

    def __init__(self):
        self.model = IsolationForest(contamination=0.02, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'transaction_amount', 'transaction_hour',
            'num_transactions_day', 'distance_from_home'
        ]

    def fit(self, data):
        X = self.scaler.fit_transform(data[self.features])
        self.model.fit(X)
        print("✅ Anomaly Detector trained successfully.")
        return self

    def detect(self, transaction: dict):
        df = pd.DataFrame([transaction])[self.features]
        X = self.scaler.transform(df)
        score = self.model.decision_function(X)[0]
        is_anomaly = self.model.predict(X)[0] == -1
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': round(score, 4),
            'status': "🚨 ANOMALY DETECTED" if is_anomaly else "✅ NORMAL"
        }


# ─────────────────────────────────────────────
# 4. MAIN — DEMO
# ─────────────────────────────────────────────
def main():
    print("="*55)
    print("  SME BANK FRAUD DETECTION SYSTEM")
    print("  Author : Saiful (saifulcu105-lang)")
    print("  Target : Community Banks & Credit Unions")
    print("="*55)

    # Generate data
    print("\n📊 Generating transaction data...")
    data = generate_sample_data(n_samples=10000, fraud_rate=0.02)
    print(f"   Total transactions : {len(data):,}")
    print(f"   Fraud cases        : {data['is_fraud'].sum():,}")
    print(f"   Legitimate cases   : {(data['is_fraud']==0).sum():,}")

    # Train fraud detector
    detector = FraudDetector()
    detector.train(data)

    # Train anomaly detector
    print("\n🔄 Training Anomaly Detector...")
    anomaly = AnomalyDetector()
    anomaly.fit(data)

    # ── Test: Legitimate transaction ──
    print("\n" + "─"*55)
    print("TEST 1 — Legitimate Transaction")
    print("─"*55)
    legit_tx = {
        'transaction_amount':   150.00,
        'transaction_hour':     14,
        'customer_age':         45,
        'account_balance':      5000.00,
        'num_transactions_day': 2,
        'distance_from_home':   5.0,
        'is_foreign':           0,
        'is_online':            0
    }
    result = detector.predict_transaction(legit_tx)
    anomaly_result = anomaly.detect(legit_tx)
    print(f"  Amount      : ${legit_tx['transaction_amount']:,.2f}")
    print(f"  Risk Score  : {result['emoji']} {result['risk_score']}/100")
    print(f"  Risk Level  : {result['risk_level']}")
    print(f"  Action      : {result['action']}")
    print(f"  Anomaly     : {anomaly_result['status']}")

    # ── Test: Fraudulent transaction ──
    print("\n" + "─"*55)
    print("TEST 2 — Suspicious Transaction")
    print("─"*55)
    fraud_tx = {
        'transaction_amount':   9800.00,
        'transaction_hour':     2,
        'customer_age':         22,
        'account_balance':      200.00,
        'num_transactions_day': 15,
        'distance_from_home':   850.0,
        'is_foreign':           1,
        'is_online':            1
    }
    result2 = detector.predict_transaction(fraud_tx)
    anomaly_result2 = anomaly.detect(fraud_tx)
    print(f"  Amount      : ${fraud_tx['transaction_amount']:,.2f}")
    print(f"  Risk Score  : {result2['emoji']} {result2['risk_score']}/100")
    print(f"  Risk Level  : {result2['risk_level']}")
    print(f"  Action      : {result2['action']}")
    print(f"  Anomaly     : {anomaly_result2['status']}")

    print("\n" + "="*55)
    print("  ✅ System ready for SME bank deployment")
    print("  🇺🇸 Protecting American community banks")
    print("="*55)


if __name__ == "__main__":
    main()

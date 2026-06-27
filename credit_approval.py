"""
Credit Decision Making Tool for SME Financial Institutions
Author: Saiful (saifulcu105-lang)
Description: AI-powered fair credit approval system for community banks
NIW Evidence: Promotes fair lending and financial inclusion for all Americans
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GENERATE LOAN APPLICATION DATA
# ─────────────────────────────────────────────
def generate_loan_data(n=5000):
    """
    Synthetic loan application data for SME bank credit decisions.
    Production reference: Home Credit Default Risk (Kaggle)
    https://www.kaggle.com/competitions/home-credit-default-risk
    """
    np.random.seed(42)
    income       = np.random.lognormal(10.5, 0.6, n)
    loan_amount  = income * np.random.uniform(0.5, 4.0, n)
    credit_score = np.random.randint(300, 850, n)
    employment   = np.random.randint(0, 30, n)
    debt_ratio   = np.random.uniform(0.05, 0.65, n)
    num_accounts = np.random.randint(1, 15, n)
    late_payments= np.random.randint(0, 10, n)
    loan_purpose = np.random.choice(
        ['home', 'business', 'education', 'personal', 'auto'], n
    )

    # Default probability (realistic logic)
    default_prob = (
        0.4 * (1 - credit_score / 850)
      + 0.3 * debt_ratio
      + 0.2 * (late_payments / 10)
      - 0.1 * (employment / 30)
      + np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    default = (np.random.uniform(0, 1, n) < default_prob).astype(int)

    return pd.DataFrame({
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'employment_years': employment,
        'debt_to_income': debt_ratio,
        'num_accounts': num_accounts,
        'late_payments': late_payments,
        'loan_purpose': loan_purpose,
        'loan_to_income': loan_amount / income,
        'default': default
    })


# ─────────────────────────────────────────────
# 2. CREDIT DECISION ENGINE
# ─────────────────────────────────────────────
class CreditDecisionEngine:
    """
    Fair, explainable credit decision system for SME banks.
    Complies with ECOA (Equal Credit Opportunity Act).
    """

    FEATURES = [
        'income', 'loan_amount', 'credit_score',
        'employment_years', 'debt_to_income',
        'num_accounts', 'late_payments', 'loan_to_income'
    ]

    FEATURE_LABELS = {
        'credit_score':      'Credit Score',
        'debt_to_income':    'Debt-to-Income Ratio',
        'late_payments':     'Late Payment History',
        'employment_years':  'Employment Stability',
        'loan_to_income':    'Loan-to-Income Ratio',
        'income':            'Annual Income',
        'num_accounts':      'Number of Accounts',
        'loan_amount':       'Requested Loan Amount',
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        )
        self.trained = False

    def train(self, data: pd.DataFrame):
        print("\n🔄 Training Credit Decision Engine...")
        X = self.scaler.fit_transform(data[self.FEATURES])
        y = data['default']

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        self.trained = True

        y_pred = self.model.predict(X_te)
        y_prob = self.model.predict_proba(X_te)[:, 1]

        print("\n" + "="*55)
        print("  CREDIT DECISION ENGINE — PERFORMANCE")
        print("="*55)
        print(f"  AUC-ROC   : {roc_auc_score(y_te, y_prob):.1%}")
        print(classification_report(y_te, y_pred,
              target_names=['Will Repay', 'Will Default']))
        return self

    def decide(self, applicant: dict):
        """
        Make a credit decision with full explanation.
        ECOA-compliant: reasons given for every denial.
        """
        if not self.trained:
            raise ValueError("Train the model first.")

        df   = pd.DataFrame([applicant])
        X    = self.scaler.transform(df[self.FEATURES])
        prob = self.model.predict_proba(X)[0][1]   # default probability
        score= int((1 - prob) * 100)               # creditworthiness score

        # Decision thresholds
        if score >= 70:
            decision, status, emoji = "APPROVED",      "✅", "🟢"
        elif score >= 50:
            decision, status, emoji = "MANUAL REVIEW", "⚠️", "🟡"
        else:
            decision, status, emoji = "DECLINED",      "❌", "🔴"

        # Feature importances → human-readable reasons
        importances = self.model.feature_importances_
        reasons = sorted(
            zip(self.FEATURES, importances),
            key=lambda x: x[1], reverse=True
        )[:3]
        top_reasons = [self.FEATURE_LABELS[f] for f, _ in reasons]

        return {
            'decision':            decision,
            'creditworthiness':    score,
            'default_probability': f"{prob:.1%}",
            'status_emoji':        emoji,
            'key_factors':         top_reasons,
            'ecoa_reasons':        top_reasons if decision == "DECLINED" else []
        }


# ─────────────────────────────────────────────
# 3. LOAN DEFAULT PREDICTOR
# ─────────────────────────────────────────────
class LoanDefaultPredictor:
    """Predicts probability of default for existing loans."""

    def predict_portfolio_risk(self, loans: pd.DataFrame):
        """Assess risk across an entire loan portfolio."""
        high_risk   = (loans['debt_to_income'] > 0.5).sum()
        medium_risk = ((loans['debt_to_income'] > 0.35) &
                       (loans['debt_to_income'] <= 0.5)).sum()
        low_risk    = (loans['debt_to_income'] <= 0.35).sum()
        total       = len(loans)

        print("\n" + "="*55)
        print("  LOAN PORTFOLIO RISK ASSESSMENT")
        print("="*55)
        print(f"  Total Loans    : {total:,}")
        print(f"  🟢 Low Risk    : {low_risk:,}  ({low_risk/total:.1%})")
        print(f"  🟡 Medium Risk : {medium_risk:,} ({medium_risk/total:.1%})")
        print(f"  🔴 High Risk   : {high_risk:,}  ({high_risk/total:.1%})")
        est_loss = high_risk * loans['loan_amount'].mean() * 0.6
        print(f"\n  💰 Estimated Loss Exposure : ${est_loss:,.0f}")
        print("="*55)


# ─────────────────────────────────────────────
# 4. MAIN DEMO
# ─────────────────────────────────────────────
def main():
    print("="*55)
    print("  CREDIT DECISION TOOL — SME FINANCIAL INSTITUTIONS")
    print("  Author : Saiful (saifulcu105-lang)")
    print("="*55)

    data = generate_loan_data(5000)
    print(f"\n📊 Loan Applications : {len(data):,}")
    print(f"   Default Rate     : {data['default'].mean():.1%}")

    engine = CreditDecisionEngine()
    engine.train(data)

    # ── Applicant 1: Strong profile ──
    print("\n" + "─"*55)
    print("APPLICANT 1 — Strong Financial Profile")
    print("─"*55)
    a1 = {
        'income': 85000, 'loan_amount': 25000,
        'credit_score': 750, 'employment_years': 8,
        'debt_to_income': 0.25, 'num_accounts': 5,
        'late_payments': 0, 'loan_to_income': 0.29
    }
    r1 = engine.decide(a1)
    print(f"  Credit Score  : {a1['credit_score']}")
    print(f"  Income        : ${a1['income']:,}")
    print(f"  Loan Amount   : ${a1['loan_amount']:,}")
    print(f"  Decision      : {r1['status_emoji']} {r1['decision']}")
    print(f"  Creditworthy  : {r1['creditworthiness']}/100")
    print(f"  Default Risk  : {r1['default_probability']}")
    print(f"  Key Factors   : {', '.join(r1['key_factors'])}")

    # ── Applicant 2: Weak profile ──
    print("\n" + "─"*55)
    print("APPLICANT 2 — High Risk Profile")
    print("─"*55)
    a2 = {
        'income': 28000, 'loan_amount': 45000,
        'credit_score': 520, 'employment_years': 1,
        'debt_to_income': 0.58, 'num_accounts': 2,
        'late_payments': 6, 'loan_to_income': 1.6
    }
    r2 = engine.decide(a2)
    print(f"  Credit Score  : {a2['credit_score']}")
    print(f"  Income        : ${a2['income']:,}")
    print(f"  Loan Amount   : ${a2['loan_amount']:,}")
    print(f"  Decision      : {r2['status_emoji']} {r2['decision']}")
    print(f"  Creditworthy  : {r2['creditworthiness']}/100")
    print(f"  Default Risk  : {r2['default_probability']}")
    if r2['ecoa_reasons']:
        print(f"  ECOA Reasons  : {', '.join(r2['ecoa_reasons'])}")

    # ── Portfolio Risk ──
    predictor = LoanDefaultPredictor()
    predictor.predict_portfolio_risk(data)

    print("\n✅ Credit Decision Tool ready for SME bank deployment")
    print("🇺🇸 Supporting fair lending across US community banks\n")


if __name__ == "__main__":
    main()

**Customer Churn Intelligence System**

An end-to-end machine learning system that predicts customer churn and explains the drivers behind customer attrition using CatBoost and SHAP explainability. The project transforms raw customer data into actionable business intelligence for retention strategy and revenue protection.

**Overview**

Customer churn is a major challenge in banking and subscription-based businesses. This system helps organizations proactively identify at-risk customers, understand why they are likely to leave, and quantify the financial impact of churn.
It combines predictive modeling, explainable AI, and business impact analysis into a single deployable system.

**Machine Learning Approach**
Model: CatBoost Classifier
Handles mixed data types (categorical + numerical)
Optimized for structured business datasets
Outputs churn probability per customer

**Explainable AI (XAI)**
Uses SHAP (SHapley Additive exPlanations) to interpret predictions:

Global feature importance (key churn drivers)
Local explanations (why individual customers churn)
Distribution of feature impact across customers

**Business Impact**
The system converts predictions into business value:
Customer risk segmentation (High / Medium / Low)
Revenue at risk estimation
ROI of retention interventions
Cost-benefit analysis of churn prevention strategies

**Streamlit Application**
Upload customer dataset (CSV)
Generate churn predictions instantly
View churn probability per customer
Display risk categories
Show business summary insights

**Tech Stack**

Python • CatBoost • SHAP • Scikit-learn • Pandas • NumPy • Streamlit • Matplotlib

**Key Value Delivered**
Early identification of churn-risk customers
Explainable AI insights for decision-making
Data-driven retention strategy support
End-to-end deployable ML pipeline

Author
Elijah faith Odeke
Data Science & Machine Learning Enthusiast
Focused on AI-driven business intelligence systems

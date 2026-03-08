# CreditRisk
This project is made to gain an understanding of WOE(Weight of Evidence) ,IV(information value),SHAP  and their application in credit risk modelling. Probability of Deafault model is made using logistic regression and ML models based on XGBoost and Random forest are also compared.
# DataSet
The dataset used is the German Credit Data from the UCI Machine Learning Repo. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
# WOE/IV
A class WOEEncoder is made that creates bins and computes WOE and IV values for all features. Care has beenn taken that fewer bins are used to catch important patterns in the data while leaving out noise. IV generally helps to rank variables on the basis of their importance, standard IV thereshold that is used are:
IV < 0.02 ->not useful for prediction
IV  0.02 to 0.1 -> weak predictive power
IV < 0.1 to 0.3 ->medium predictive power
IV < 0.3 to 0.5 ->strong predictive power
IV >0.5 -> Suspicious
Features with IV > 0.02 are selected and WOE-transformed as inputs to the Logistic Regression model. Numeric features are automatically binned using quantile-based binning before WOE encoding
# Model Training 
3 models are trained and evaluated. Train and test split is 80% and 20% respectively.
1) Logistic Regression-It is trained on WOE transformed data. It is the baseline model
2) Random Forest-200 trees are trained on label-encoded features. It is used to capture non-linear interactions and for feature importance comparison.
3) XGBoost-300 rounds of gradient boosting with subsampling. It achieved the highest predictive performance.
# Model Validation and Model Metrics
Each model is evaluated on the 80/20 train/test split using the following metrics:
1) Area under the ROC Curve(AUC) - Basically measure how well the model ranks risky customers higher than the safe customers.(target: > 0.70)
2) KS Statistic: maximum separation between good and bad score customers (target: > 0.30)
3) GINI - directly obtained using AUC (GINI = 2 × AUC − 1).Higher GINI value means that the model is better in distinguishing risky customers.(target: > 0.40)
# SHAP
SHAP values are computed for all models to explain individual predictions:
1) Bar plot- mean absolute SHAP value per feature (global importance)
2) Beeswarm plot- distribution of SHAP values across all test samples
3) Waterfall plot - single borrower explanation for a high-risk case (PD > 0.60
# RESULTS
1) IV Analysis:
   <img width="2019" height="914" alt="image" src="https://github.com/user-attachments/assets/90a76ad6-3b93-470f-a96e-f79106cd1064" />
2) WOE: <img width="1934" height="744" alt="image" src="https://github.com/user-attachments/assets/4d55acf3-5a0b-481b-99a6-e35ae1bbe68d" />

3) Model Performance: XGBoost has higher score on the AUC and thus GINI metrics while Random Forest has higher KS Statistics score.<img width="2084" height="889" alt="image" src="https://github.com/user-attachments/assets/cdcf67b0-5c2e-42d9-98d6-c374322bba23" />

4) PD Score Distribution: <img width="2230" height="740" alt="image" src="https://github.com/user-attachments/assets/676182fe-ff2a-4ffb-a7ec-2cee20cc4584" />
Logistic regression does not have perfect seperation while random forest has better seperation with overlap region being smaller. XGBoost pushes bad custoers to higher PD and thus has the strongest seperation

5) SHAP:<img width="1351" height="885" alt="image" src="https://github.com/user-attachments/assets/fae58b32-94c7-48a2-9e36-fd279058b752" />
<img width="1376" height="885" alt="image" src="https://github.com/user-attachments/assets/fbd791e9-a7be-4291-8363-2b3b71dc55fe" />
<img width="1356" height="768" alt="image" src="https://github.com/user-attachments/assets/e93cb2af-9f5d-4fa1-a5af-fb6b6b2e87ba" />
<img width="1429" height="922" alt="image" src="https://github.com/user-attachments/assets/8b73c08e-6d89-4a6d-a552-1adb47a54239" />




6) Feature Importance:<img width="2086" height="887" alt="image" src="https://github.com/user-attachments/assets/6a85bb6d-87b9-49a5-a304-926794966473" />

7) Confusion Matrix:<img width="2211" height="740" alt="image" src="https://github.com/user-attachments/assets/24cdc739-c039-426b-9096-a1c1e5c0976c" />




# Project Title

### Capstone Project - Mental Illness Risk prediction (**MUTI-CLASS CLASSIFICATION USE CASE**)

## Notebook and other links are in "Outline of project" section

# Executive summary

1. This project develops an interpretable machine learning framework that evaluates clinical and behavioral data to predict the risk of mental health conditions. By identifying high-risk individuals early, this tool empowers clinicians and patients to implement proactive, personalized interventions before acute crises occur

2. At the sametime this framework should NOT miss any low/medium RISK patients who can develop mental illness in future

3. Evaluation Metrics: Evaluated primarily on Recall, F1-Score, and AUC-ROC to minimize false negatives( MAXIMIZE recall ) and ensure maximum safety in medical triage.

4. Mental Risk Prediction is a **multi-class classification** Use Case where a patient can be categorized as 'Low Risk', 'Medium Risk' or 'High Risk' depending on their socioeconomic factors , clinical factors & past treatment/medical history.

# Rationale


1. **For Patients:** Shifts the focus toward prevention rather than just symptom management. Early risk detection allows individuals to seek support before conditions escalate, improving long-term quality of life

2. **For Clinicians:** Acts as a powerful clinical decision-support tool. By automatically analyzing vast datasets like Electronic Health Records, algorithms help doctors identify subtle warning signs that might be missed during brief routine checkups

3. Identifying mental illness risk via machine learning transforms psychiatric care from reactive to proactive.

# Research Question


    - How can I identify and predict Mental Health Risk factors which is a serious global challenge ? 
    - The  conditions like  Anxiety and Depression are the most common disorders globally and saw a massive surge in prevalence over the past three decades

# Data Sources
What data will you use to answer you question?

    - I used Kaggle data source: https://www.kaggle.com/datasets/guriya79/mental-health-disorder

# Methodology
What methods are you using to answer the question?

I will use following techniques:

    1. Class imbalance analysis
    2. Data skewness check
    3. PCA 
    4. Correlation Matrix analysis
    5. Encoding input features
    6. Different multi-class classification algorithm comparison
    7. ROC-AUC analysis
    8. Precision, Recall & F1 score comparison  to identify which algorithm produces better results for this Use Case.

# Findings / Results

**Main Reference notebook : analysis_with_feature_engg.ipynb**

**Phase - EDA**

    - This UseCase is on mental illness. I don't see any missing medical/clinical information for each patient in the given dataset. This is good in one way that I don'tneed to  use mean/median/mode or any other technique to fill-in the NULLS. The mdeical/clinical fators are very specific to a person, should not be generalized. 
    - Incontrast I see data is noisy like for example, one member's anxiety_score,depression_score,stress_level are 4,4,5 respectively. If we feed raw data with less variance model will not learn patterns correctly and might show high accuracy score during training, which is incorrect.
    - Mental health risks are rarely the result of a single isolated factor, but rather combinations of lifestyle and environment. So feature enginnering is an important step
    - Transforming this raw data into meaningful features highlights hidden patterns, model accuracy, and reduces dimensionality so machine learning algorithms can better capture complex psychological and lifestyle interactions.
    - Generating new features ( derived columns/feature ) uncover hidden correlations that raw features miss.
    - Engineer features that highlight anomalies (like combining multiple high stress and anxiety indicators) to help models better delineate at-risk individuals.
    - Data skewness observed. This needs to be handled by logrithmic transformation before feeding into an algorithm.
    - Target class imbalance observed. Use/choose proper algorithm.
    - PCA analysis indicates both categorical & numerical features play important role

**Phase - MODELLING**

Explanation of Few Metrics that I will be using to evaluate the model performances

    1. Recall - Use Recall when the cost of a false negative is high (e.g., medical diagnosis or fraud detection, where missing a sick patient or a fraudulent charge is dangerous)
    2. Precision -  when the cost of a false positive is high (e.g., spam filters, where marking a good email as spam is bad).
    3. F1-score - Use F1 Score when you need a balance between precision and recall, particularly when dealing with skewed class distributions
    4. Accuracy - Use Accuracy when your dataset has a balanced distribution of classes and false positives/false negatives carry similar weight.

As my use case is from HealthCare domain to identify patients with high/medium/risk , my goal is to improve on recall at the cost of precision. This will ensure my model does not miss out patients who are at high to medium , or even at low risk. **If I  missout to catch a low risk patients today, he/she might develop more illness and transion to medium to high risk.** Hence identifying patients with potential RISK **at earliy stage** will help doctors to start proper treatments at early stage.  

To find initial base model I used four classification algorithms, namely LogisticRegression,DecisionTree, KNearestNeighbor, SupportVectorMachine

    - Out of 4 classifiers, the performance & other metrics are  poor for KNN/DecisionTree/SVM(refer my notebook). 
    - Hence, from overall performace(Train/Test time) and recall/precision/F1-score perspective I will take **LogisticRegression as my base model** & look for other algorithm that can handle class imbalance & non linear relationships in a better way.  

**LogisticRegression Classification Report(Base Model): without class weight**

              precision    recall  f1-score   support

    Low Risk       0.64      0.58      0.61      1871
 Medium Risk       0.57      0.71      0.63      2365
   High Risk       0.65      0.30      0.41       764

    accuracy                           0.60      5000
   macro avg       0.62      0.53      0.55      5000
weighted avg       0.61      0.60      0.59      5000

This indcates recall for **high_risk class is very low(0.3)**. Even the accuracy score is 0.6. For this Use Case I need to improve the recall score

***Use robust algorithms to compare if those produce better result compared to BASE model***

Following algorithms compared with Base Model

        **1. RandomForest**
        **2. XGBoost Classifier**
        **3. LogisticRegression**
        **4. Neural network**


- Create WEIGHTED_SAMPLE_CLASS to handle - HIGH_RISK(most imbalanced class in dataset) & LOW_RISK (moderately imbalance). This is necessary to handle class imbalance found in  EDA phase  

**WEIGHTED_SAMPLE_CLASS approach has handled class imbalance of HIGH_RISK & LOW_RISK classes which has improved the Recall score. This was needed to make sure I don't miss any HIGH and LOW RISK patient**

**Metrics after applying WEIGHTED_SAMPLE_CLASS - Result & Analysis**

***LogisticRegression***

Classification Report:
              precision    recall  f1-score   support

    Low Risk       0.62      0.67      0.64      1871
 Medium Risk       0.57      0.38      0.46      2365
   High Risk       0.38      0.70      0.49       764

***RandomForest***
Classification Report:
              precision    recall  f1-score   support

    Low Risk      0.65      0.57      0.61      1871
 Medium Risk      0.56      0.73      0.63      2365
   High Risk      0.61      0.19      0.29       764

***XGBoost***
Classification Report:
              precision    recall  f1-score   support

    Low Risk      0.62      0.68      0.65      1871
  Medium Risk     0.60      0.43      0.50      2365
    High Risk     0.41      0.65      0.50       764


***outcome:***
    - With weighted_samlple class high_risk & low_risk class **recall scores improved significantly for LogisticRegression & XGBoost**.


**Confusion Matrix - Result & analysys**

    - With CLASS_SAMPLE_WEIGHT the low & high risk prediction number has improved a lot for XGBoost & LogisticRegression

    Example : The output under 'Confusion Matrix Comparison on base pipelines before applying gridsearch optimization'  section in indicates 

    XGBoost shows a balanced performance compared to RandomForest & LogisticRegression. It has a slightly better True Positive count for Low Risk than Logistic Regression and better performance for High Risk compared to Random Forest. The Medium Risk class remains challenging, with a substantial portion misclassified. 


**GridSearch - Hyperparameter tuned models - Result & Analysis**

***XGBoost continues to show slightly better performance with its optimized hyper-parameters compared to RandomForest & LogisticRegression.***

Best parameters for XGBoost: {'model__subsample': 0.8, 'model__sample_weight': None, 'model__n_estimators': 300, 'model__max_depth': 3, 'model__learning_rate': 0.1, 'model__gamma': 0.2, 'model__colsample_bytree': 0.8} Best F1-macro score for XGBoost: 0.5651085914668911

***Explanation:***

- Tree Structure and Size :

    model__max_depth: 3 — Limits each tree to a maximum depth of three splits. This creates "stumps" or shallow trees, which prevent the model from memorizing noise in the training data and reduce overfitting.

    model__n_estimators: 300 — Sets the total number of boosting rounds or trees built sequentially. Combined with a slower learning rate, 300 trees allow the model to refine its predictions gradually.

- Learning and Step Size

    model__learning_rate: 0.1 — Shrinks the contribution of each new tree by 10%. A lower rate means the model learns slower and requires more trees, but it results in a more robust and stable final predictor.


- Regularization and Randomness

    model__subsample: 0.8 — Uses 80% of the training rows at random to build each tree. This stochastic sampling adds diversity and prevents overfitting.

    model__colsample_bytree: 0.8 — Selects 80% of the features randomly for each tree. It forces the trees to look at different subsets of data columns, reducing feature dominance.

    model__gamma: 0.2 — Requires a minimum loss reduction of 0.2 to make a further partition on a leaf node. This acts as a strict gatekeeper, stopping the model from making unnecessary or overly specific splits.

- Data Weights

    model__sample_weight: None — Treats all training rows with equal importance. No special class balancing or custom weight scaling is applied during trainin



**Precision vs Recall plot at (0.1,0.2,0.3,0.4 & 0.5) Threshold - Result and Analysis**

What is Threshold? 

    In machine learning, a classification threshold (or decision threshold) is a number used to turn a model's continuous probability output into a clear "yes" or "no" choice. The default value is usually 0.5, meaning scores above 0.5 mean a positive class, but changing this line alters model errors and accuracy balance.

Why Adjust the Threshold?

    Default 0.5 Is Not Always Best: Real-world data often has uneven costs for mistakes.

    High Threshold (> 0.5): Makes the model strict. It reduces **false positives (wrong alarms) but causes more false negatives (missed events)**. Example: Content Moderation.

    Threshold (< 0.5): Makes the model loose. It **catches more true items (higher recall) but creates more false alarms (lower precision)**. Example: Use this for medical tests where missing a disease is dangerous.

In my Use Case I need higher recall for "high & low risk" patient categories. So this validation targets to findout best Threshold at which the model should operate.  

    - XGBoost demonstrates a **good balance** between recall and precision, especially when optimizing for F1-score **across different classes**. It generally provides competitive performance compared to other two models

    - Logistic Regression can achieve very high recall for 'Low Risk' and 'Medium Risk' at low thresholds, similar to Random Forest. When optimizing for precision, it can be quite effective for the 'High Risk' class, but like other models, this comes with a trade-off in recall.

***In this scenario a threshold between 0.30 and 0.40 would be best to keep balance between Recall & Precision.***


**Misclassification - Result & Analysis**

    - Misclassification Percentage Comparison plot: RandomForest, XGBoost, and LogisticRregression 

***Results indicate XGBoost has **lowest** misclassification rate***

**Feature importance comparison of LogisticRegression/RandomForest/Xgboost - Result & Analysis**

    - The plot indicates RandomForest rely mostly on 2 features active_screen_ratio and protective_buffer_score. This might create problem as its NOT puting weightage on other paramaters.

    - LogisticRegression & XGBoost uses more features to make predictions, which is good.
    - Out of LogisticRegression & XGBoost, **XGBoost** uses more features and more evenly distributes weightage across different features.


**Neural Network - Result & Analysis-->> Reference notebook : analysis_with_feature_engg_NeuralNetwork.ipynb**

Classification Report:
               precision    recall  f1-score   support

    Low Risk       0.66      0.53      0.59      1871
 Medium Risk       0.56      0.67      0.61      2365
   High Risk       0.48      0.42      0.45       764

This Metrics are close to LogisticRegression/XGBoost  but NOT better. So **Neural Network didn't add any extra value.**

**Testing with UNSEEN  SYNTHETIC data - Using the saved tuned models -->> Reference notebook : TestSavedModel.ipynb**

    - I see the the performace has degraded. Need to see why? 
    - With synthetic data max recall_score is < 0.423 which is way less than untuned XGBoost/LogisticRegression model. 

**SYNTHETIC DATA quality Analysis**

    - Kolmogorov-Smirnov (KS-test) indicates: 15 out of 21 numerical columns show a significant difference in distribution between original and synthetic data (p < 0.05).
    This suggests that the synthetic data's ***distributions for these columns are statistically different from the original data, which can impact model performance.***
    - NULL HYPOTHESIS testing
    15 out of 21 columns have p-values of 0.0000, strongly **rejecting the null hypothesis that their distributions are the same**

***Outcome***
    - Try different ways to generate SYNTHETIC data as next step/phase
    - HeatMap Analysis of SYNTHETIC DATA vs Original Data

***Outcome***

***This comparison highlights a critical issue: the synthetic data fails to preserve the intricate correlation structure of the original dataset. While the original data shows clear relationships between various mental health indicators and lifestyle factors, the synthetic data largely lacks these meaningful correlations.This proves the poor metrics are due to SYNTHETIC Data quality issue. This SYNTHETIC data is not a reliable source. I need to look for other similar source/algorithm to generate SYNTHETIC data***


**Overall Verdict**
1. Based on the above analysis and results for this UseCase

Considering the following factors
        a. Overall performance (train/test time)
        b. Metrics (Recall, F1-score, Precision, Confusion Metrix, ROC-AUC)
        c. Robustness in feature selection/utilization
        d. Misclassification rate
        e. Consistent precision/Recall balance across different THRESHOLDs 

 My vote 
            I. **Choice 1 - XGBOOST (Can be productionized )**
            II. Choice 2 - LogisticRegression (Can be used as baseline model for further analysis)
2. To improve model's Recall score for high_risk/low_risk keep threshold between 0.3 to 0.4. 

# Next steps

Actionable Items :

    1. Analyse which other Clinical Assessment, Diagnostic Criteria, Identifying Triggers can be used as additional input features to identify mental risk factors accurately. 
    2. Talk to healthcare domain expert & get more insights on clinical data.This will help in generating synthetic data with proper distribution. My data generation technique didn't produce good reliable data for testing.
    3. Discuss with domain experts/ healthcare providers to see how/from where I can get real data
    4. When I get access to real data stratetize how to  handle tokenized PI/SPI data elements, as appropriate.  
    5. Additional Feature Engineering Techniques, if needed based on domain experts'feedback,  to Enhance Class Separation
    6. Interaction Features

# Outline of project

## Repository Structure

```text
.
├── saved_model/                    # Hyperparameter tuned model files saved for any future reference
├── data/                           # Source datafiles(Kaggel) for analysis/modeling  & Unseen synthetic data for testing the model  
├── src/                            # Notebooks: 1. (EDA + Base models + RandomForest+XGBoost) 2. Test saved models with unseen synthetic data 3. Neural Network
│   ├── src_synthetic_data/         # One python(.py)-Generate synthetic data 
└── README.md       # Project overview
```



# Contact and Further Information

Will be filled up later during final submission
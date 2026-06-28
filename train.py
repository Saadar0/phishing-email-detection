# train.py
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix
)

from utils import TextPreprocessor

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
df1 = pd.read_csv('data/CEAS_08.csv/CEAS_08.csv')
df2 = pd.read_csv('data/SpamAssasin.csv/SpamAssasin.csv')
df3 = pd.read_csv('data/Nigerian_Fraud.csv/Nigerian_Fraud.csv')
df4 = pd.read_csv('data/Phishing_Email.csv/Phishing_Email.csv')[['Email Text', 'Email Type']]

df_group_a = pd.concat([df1, df2, df3])[['body', 'label']]
df_group_a.columns = ['Email Text', 'Email Type']

df4['Email Type'] = df4['Email Type'].apply(
    lambda x: 1 if 'phishing' in str(x).lower() else 0
)

phishing_df = pd.concat([df_group_a, df4], ignore_index=True)
phishing_df = phishing_df.dropna().reset_index(drop=True)

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
tp = TextPreprocessor()
phishing_df['processed_text'] = tp.transform(phishing_df['Email Text'])

num_features_list = [
    tp.extract_features(cleaned, original)
    for cleaned, original in zip(phishing_df['processed_text'], phishing_df['Email Text'])
]

df_numerical = pd.DataFrame(num_features_list)
numerical_cols = list(df_numerical.columns)

df_combined = pd.concat([phishing_df[['processed_text', 'Email Text', 'Email Type']], df_numerical], axis=1)
X = df_combined[['processed_text', 'Email Text'] + numerical_cols]
y = df_combined['Email Type']

# =============================================================================
# 3. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# =============================================================================
# 3.1 SAVE TEST DATA FOR EVALUATION
# =============================================================================
# We create a specific DataFrame for the test by combining X_test and y_test
test_export = X_test.copy()
test_export['Email Type'] = y_test

# Save as CSV
test_export.to_csv('test_data_split.csv', index=False)
print("✅ Fichier 'test_data_split.csv' sauvegardé avec succès.")
# =============================================================================
# 4. FEATURE SELECTION
# =============================================================================

print("\n🔍 Cleaning and selecting numerical features...")

# Step A: Remove features that are constant (zero variance)
vt = VarianceThreshold(threshold=0)
X_train_filtered = vt.fit_transform(X_train[numerical_cols])

# Get the names of features that survived the variance check
remaining_cols = X_train[numerical_cols].columns[vt.get_support()].tolist()
print(f"Dropped {len(numerical_cols) - len(remaining_cols)} constant features.")

# Step B: Select the top 80% based on F-score
selector = SelectPercentile(f_classif, percentile=80)
selector.fit(X_train_filtered, y_train)

# Identify the final set of features to keep
selected_features = [remaining_cols[i] for i in selector.get_support(indices=True)]
print(f"✅ Selected {len(selected_features)} features (Percentile: 80%)")

X_train_num = X_train[selected_features]
X_test_num  = X_test[selected_features]

# =============================================================================
# 5. COLUMN TRANSFORMER
# =============================================================================
word_tfidf = TfidfVectorizer(
    max_features=7000, min_df=3, max_df=0.90,
    ngram_range=(1, 3), stop_words='english', sublinear_tf=True,
    token_pattern=r'\b\w+\b|\[url\]|\[email\]|\[phone\]|\[number\]'
)

char_tfidf = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5), max_features=2000, sublinear_tf=True, min_df=3
)

# Numerical pipeline: Handle outliers THEN scale to [0,1] for Naive Bayes
num_pipe = Pipeline([
    ('robust', RobustScaler()),
    ('minmax', MinMaxScaler())
])

col_preprocessor = ColumnTransformer(
    transformers=[
        ('word', word_tfidf, 'processed_text'),
        ('char', char_tfidf, 'processed_text'),
        ('num',  num_pipe, selected_features)
    ]
)

print("Fitting ColumnTransformer...")
X_train_t = col_preprocessor.fit_transform(pd.concat([X_train[['processed_text']], X_train[selected_features]], axis=1))
X_test_t = col_preprocessor.transform(pd.concat([X_test[['processed_text']], X_test[selected_features]], axis=1))

# =============================================================================
# 6. MODELS
# =============================================================================
clf_nb  = ComplementNB(alpha=0.1)

clf_rf  = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=5,
    class_weight='balanced', n_jobs=-1, random_state=42
)

clf_lr  = LogisticRegression(
    max_iter=1000, class_weight='balanced', C=0.5, solver='saga'
)

clf_svm = CalibratedClassifierCV(
    SGDClassifier(loss='hinge', max_iter=1000, random_state=42, class_weight='balanced')
)

voting_clf = VotingClassifier(
    estimators=[('nb', clf_nb), ('rf', clf_rf), ('lr', clf_lr), ('svm', clf_svm)],
    voting='soft', weights=[1, 4, 2, 3]
)

print("Training Voting Classifier...")
voting_clf.fit(X_train_t, y_train)

# =============================================================================
# 7. EVALUATION
# =============================================================================
y_proba = voting_clf.predict_proba(X_test_t)[:, 1]

print("\nTHRESHOLD ANALYSIS (FN cost = 100x FP cost)")
fp_cost, fn_cost = 1, 10
best_thresh, best_cost = 0.5, float('inf')

for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    yp = (y_proba >= thresh).astype(int)
    tn, fp, fn, true_p = confusion_matrix(y_test, yp).ravel()
    cost = fp * fp_cost + fn * fn_cost
    if cost < best_cost:
        best_cost, best_thresh = cost, thresh
    print(f"Thresh: {thresh:.2f} | FN: {fn:>4} | FP: {fp:>4} | Cost: {cost:>6}")

print(f"\n💡 Recommended threshold: {best_thresh}")
y_pred = (y_proba >= best_thresh).astype(int)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Save artifacts
joblib.dump(tp, 'text_preprocessor2.pkl')
joblib.dump(col_preprocessor, 'col_preprocessor2.pkl')
joblib.dump(voting_clf, 'phishing_voting_model2.pkl')
joblib.dump(selected_features, 'selected_features2.pkl')
joblib.dump(best_thresh, 'best_threshold2.pkl')
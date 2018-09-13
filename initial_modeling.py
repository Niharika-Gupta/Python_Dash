# -----------------------------------------------------------------------------------------------
# Author: cgarcia
# Creation Date: 5.31.2018
# About: This provides a quick first look at theGeico Fraud data. Part I looks at univariate
#        feature importance by separating into positive and negative instances
#        for each target, and then ranking by p-values. Part II builds and tunes
#        3 models tolerant of high-dimensional feature sets and compares the results
#        on each target.
# -----------------------------------------------------------------------------------------------


import pandas as pd
import statsmodels.stats.weightstats as sm
import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import sys


predictions = dict()
# Read in data
dd_path = r'datadictionary_fraud_freetrial.csv'
data_path = sys.argv[1]
dd = pd.read_csv(dd_path)
data = pd.read_csv(data_path)
all_numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_data = data.select_dtypes(include=all_numeric_types)

# Extract feature and target columns
feature_df = dd[(dd['is_feature'] == '1')]
targets = [x for x in list(feature_df['Field']) if x.startswith('TARGET_')]
feature_roots = list(set(list(feature_df['Field'])).difference(targets))
all_features = [x for x in list(data) for y in feature_roots if x.startswith(y)]
numeric_features = [x for x in all_features if x in list(numeric_data)]

# PART I: BASIC UNIVARIATE EXPLORATORY ANALYSIS
#   Split into positive and negative sets by target variable. Then for each numeric feature,
#   look at the p-value of the separation (by a z-test). P-value is used as a relative measure
#   of effect size.
print('------------- INITIAL EXPLORATORY (UNIVARIATE) ANALYSIS --------------')

features = {}

for target in targets:
	t_data = data[numeric_features + [target]]
	pos = t_data[t_data[target] == 1]
	neg = t_data[t_data[target] == 0]

	ranked_features = []
	print("\n Z-TEST COMPARISON FOR TARGET = " + target + ' (SORTED BY P-VALUE)')
	ranked_features = []
	for f in numeric_features:
		try:
			_, pval = sm.ztest(list(pos[f]), list(neg[f]))
			if not(np.isnan(pval)):
				ranked_features.append((f, pval))
		except:
			pass

	# Sort ranked_features by ascending p-value
	ranked_features = sorted(ranked_features, key=lambda x: x[1])
	for x in ranked_features:
		print('    ' + str(x))

	# Set the features for the given target to those that cause no problems.
	features[target] = [x[0] for x in ranked_features]


# PART II: BASIC MODELING TO DETECT STRENGTH OF SIGNAL.
#   Build a tuned voting ensemble using three algorithms capable of high-dimensional feature sets:
#   Adaboost, Random Forest, and Gradient Boosting. This is to see if we can get a reasonable signal
#   on the initial dataset. Quick and dirty modeling approach.
print('------------- BASIC MODELING ANALYSIS --------------------------------')

random_state = 4 # Used for reproducibility of results.

for target in targets:
	data_x = data[features[target]]
	data_y = data[target]

	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = random_state)

	print("\n BUILDING MODELS FOR TARGET = " + target)

	# Make the models.
	m1 = svm.SVC()
	m2 = ensemble.RandomForestClassifier()
	m3 = ensemble.AdaBoostClassifier()

	# Set up params for grid search on the models.
	pg1 = {'C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0]}
	pg2 = {'n_estimators':[50, 100, 250, 500],
		   'max_depth': [((x / 4.0) * len(features[target])) for x in [1,2,3,4]],
		   'random_state':[random_state]}
	pg3 = {'n_estimators':[50, 100, 250, 500],
		   'random_state':[random_state]}

	# Build and score the models.

	models = zip([m1, m2, m3], [pg1, pg2, pg3], ['SVM', 'RF', 'ADABOOST'])
	for m, params, name in models:
		tuned_mod = GridSearchCV(estimator=m, param_grid=params, cv=5, scoring='roc_auc')
		tuned_mod.fit(x_train, y_train)
		preds = tuned_mod.predict(x_test)
		if name=="RF":
			preds_save = tuned_mod.predict(pd.concat([x_train,x_test]))
			predictions[target] = preds_save
		print(predictions)
		print('  RESULTS FOR MODEL ' + name + ':')
		print('    Precison: ' + str(precision_score(y_test, preds)))
		print('    Recall: ' + str(recall_score(y_test, preds)))
		print('    F1: ' + str(f1_score(y_test, preds)))
		print('    ROC AUC: ' + str(roc_auc_score(y_test, preds)))
print("predictions filled")
print(predictions)
random_forest_predictions = pd.DataFrame.from_dict(predictions)
random_forest_predictions.to_csv("predictions.csv", index= False)

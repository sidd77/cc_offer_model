import pandas
from sqlalchemy import create_engine

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, precision_recall_curve, auc

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot

from sklearn.utils import class_weight
import numpy as np

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as gbc

connstr = "postgresql://siddhartha:Pw0fwls2@cs-dubnium-imdb-prod-redshift.cviykjlhpudb.eu-west-2.redshift.amazonaws.com:5439/prd"
engine = create_engine(connstr)

conn = engine.connect()
conn.begin()

df_1 = pandas.read_sql("Select distinct a.* from analytics.sa_card_modeldata a where clicked=0;" , conn)
df_2 = pandas.read_sql("Select a.* from analytics.sa_card_modeldata a where clicked=1;" , conn)

###############################
####### Model part ############
###############################

# if len(df_1)> 5 * len(df_2):
#     df_1 = df_1.sample(5* len(df_2))


df_sample = df_1.append(df_2)
df_sample.dropna(inplace=True)

print(df_sample.shape)

# temp_set = df_sample[['rank1', 'clicked']].groupby('rank1').sum()
#
# temp_set = temp_set[temp_set['clicked'] >0]
#
# df_sample = df_sample[df_sample['rank1'] in temp_set['rank1']]
#
# print(df_sample.shape)

X = df_sample.drop(['clicked'], axis=1)
y = df_sample['clicked']
X = pandas.get_dummies(X)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(score_func=chi2, k=10).fit_transform(X,y)

# print(featureScores.nlargest(10,'Score'))  #print 10 best features

# top10 = featureScores.nlargest(10, 'Score')['Specs'].to_list()
#
# X = X[top10]

X_train, X_test, y_train, y_test  = train_test_split(X_new, y, test_size=.25)
# model = LogisticRegression(class_weight='balanced')

# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train),
                                                 # y_train)


model = gbc()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc_roc = metrics.roc_auc_score(y_test, y_pred_proba)

#
# this_model = [model, set(X), X.columns]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

auc_pr = auc(recall, precision)

pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()

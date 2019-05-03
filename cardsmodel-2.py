import pandas
from sqlalchemy import create_engine

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as gbc

connstr = "postgresql://siddhartha:Pw0fwls2@cs-dubnium-imdb-prod-redshift.cviykjlhpudb.eu-west-2.redshift.amazonaws.com:5439/prd"
engine = create_engine(connstr)

conn = engine.connect()
conn.begin()

df_1 = pandas.read_sql("Select distinct a.* from analytics.sa_cards_data a where clicked=0;" , conn)
df_2 = pandas.read_sql("Select a.* from analytics.sa_cards_data a where clicked=1;" , conn)

###############################
####### Model part ############
###############################

df_sample = df_1.append(df_2)
df_sample.dropna(inplace=True)

X = df_sample.drop(['clicked'], axis=1)
y = df_sample['clicked']
X = pandas.get_dummies(X)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pandas.DataFrame(fit.scores_)
dfcolumns = pandas.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pandas.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

# import seaborn as sns
# corrmat = df_sample.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(df_sample[top_corr_features].corr(),annot=True,cmap="RdYlGn")


top10 = featureScores.nlargest(10, 'Score')['Specs'].to_list()

X = X[top10]

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.25)
# model = LogisticRegression()
model = gbc()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#
# this_model = [model, set(X), X.columns]
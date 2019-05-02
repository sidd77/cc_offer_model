import pandas
from sqlalchemy import create_engine

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import pickle
import numpy as np
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression


connstr = "postgresql://siddhartha:Pw0fwls2@cs-dubnium-imdb-prod-redshift.cviykjlhpudb.eu-west-2.redshift.amazonaws.com:5439/prd"
engine = create_engine(connstr)

conn = engine.connect()
conn.begin()

df_1 = pandas.read_sql("Select distinct apr, score, eligibility, rank, category, pop, clicked from analytics.sa_cardmodel_data where clicked=0;" , conn)
df_2 = pandas.read_sql("Select apr, score, eligibility, rank, category, pop, clicked from analytics.sa_cardmodel_data where clicked=1" , conn)

df_1.dropna(inplace=True)
df_2.dropna(inplace=True)

plt_data = df_2.filter(['rank', 'clicked']).groupby('rank').sum()
plt_data.sort_values('rank', ascending=True)
plt_data.reset_index(inplace=True)

plt.plot(plt_data['rank'], plt_data['clicked'])

plt.xlabel('rank')
plt.ylabel('clicks')

plt.title('clicks by rank')

plt.savefig('rank_cards.png')
plt.show()

plt_data = df_2.filter(['category', 'clicked']).groupby('category').sum()

plt_data=plt_data.sort_values('clicked', ascending=False)
plt_data.reset_index(inplace=True)

plt.plot(plt_data['category'], plt_data['clicked'])

plt.xlabel('category')
plt.ylabel('clicks')

plt.xticks(rotation='vertical')

plt.title('clicks by category')
plt.savefig('category_cards.png', bbox_inches='tight')
plt.show()

plt_data = df_2.filter(['pop', 'clicked']).groupby('pop').sum()

plt_data=plt_data.sort_values('clicked', ascending=False)
plt_data.reset_index(inplace=True)

plt.plot(plt_data['pop'], plt_data['clicked'])

plt.xlabel('pop')
plt.ylabel('clicks')

plt.xticks(rotation='vertical')

plt.title('clicks by pop')
plt.savefig('pop_cards.png', bbox_inches='tight')
plt.show()

plt_data = df_2.filter(['eligibility', 'clicked']).groupby('eligibility').sum()

plt_data=plt_data.sort_values('eligibility')
plt_data.reset_index(inplace=True)

plt.plot(plt_data['eligibility'], plt_data['clicked'])

plt.xlabel('eligibility')
plt.ylabel('clicks')

plt.title('clicks by eligibility')
plt.savefig('elg_cards.png')
plt.show()

plt_data = df_2.filter(['apr', 'clicked']).groupby('apr').sum()

plt_data=plt_data.sort_values('apr')
plt_data.reset_index(inplace=True)

plt.plot(plt_data['apr'], plt_data['clicked'])

plt.xlabel('apr')
plt.ylabel('clicks')

plt.xticks(rotation='vertical')

plt.title('clicks by apr')
plt.savefig('apr_cards.png')
plt.show()

plt_data = df_2.filter(['score', 'clicked']).groupby('score').sum()

plt_data=plt_data.sort_values('clicked')
plt_data.reset_index(inplace=True)

plt.plot(plt_data['score'], plt_data['clicked'])

plt.xlabel('score')
plt.ylabel('clicks')

plt.title('clicks by score')
plt.savefig('score_cards.png')
plt.show()

# plt_data = df_2.filter(['issuer', 'clicked']).groupby('issuer').sum()
#
# plt_data=plt_data.sort_values('clicked', ascending=False)
# plt_data.reset_index(inplace=True)
#
# plt.plot(plt_data['issuer'], plt_data['clicked'])
#
# plt.xlabel('issuer')
# plt.ylabel('clicks')
#
# plt.xticks(rotation='vertical')
#
# plt.title('clicks by issuer')
# plt.savefig('issuer_cards.png', bbox_inches = 'tight')
# plt.show()


###############################
####### Model part ############
###############################



df_sample = df_1.append(df_2)

print(df_sample.shape)

df_sample.dropna(inplace=True)

print(df_sample.shape)

df_sample.reset_index(inplace=True)

X = df_sample[['rank', 'score', 'eligibility', 'apr', 'category', 'pop']]

y = df_sample['clicked']

category= pandas.get_dummies(X.category)

score = pandas.get_dummies(X.score)

pop= pandas.get_dummies(X['pop'])

# issuer = pandas.get_dummies(X.issuer)


X.drop(['category', 'score', 'pop'], axis=1, inplace=True)

X = pandas.concat([X, category, score, pop], axis=1)

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.25)


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

acc_score = accuracy_score(y_test, y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)


this_model = [model, set(X), X.columns]

pickle.dump(this_model, open('card_model', 'wb'))



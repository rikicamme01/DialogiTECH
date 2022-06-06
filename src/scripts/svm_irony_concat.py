import sys
import os 
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
from ast import literal_eval
from models.bert_rep import BertRep
from loggers.neptune_logger import NeptuneLogger
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import neptune.new as neptune


if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

logger = NeptuneLogger()

train_df = pd.read_csv(sys.argv[1] + 'data/processed/sentipolc/subj_rep_train.csv', converters={'rep': literal_eval, 'spans': literal_eval})

print('Computing embeddings..')

bert_rep = BertRep()
train_df['hs'] = train_df['spans'].map(bert_rep.four_last_hidden_state_concat).values.tolist()

test_df = pd.read_csv(sys.argv[1] + 'data/processed/sentipolc/subj_rep_test.csv', converters={'rep': literal_eval, 'spans': literal_eval})
test_df['hs'] = test_df['spans'].map(bert_rep.four_last_hidden_state_concat).values.tolist()


print('-------------------IRONY------------------------')
X_train = train_df['hs'].to_list()
y_train = train_df['iro'].to_list()

X_test = test_df['hs'].to_list()
y_test = test_df['iro'].to_list()


param_grid = {'C': [0.1, 0.5, 1, 10],
              'gamma': ['scale', 1, 0.1, 0.01],
              'degree': [3, 4, 5],
              'kernel': ['rbf','poly']}


grid = GridSearchCV(svm.SVC(class_weight = 'balanced'), param_grid, refit = True, verbose = 3, scoring=make_scorer(f1_score, pos_label=1), cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

y_pred = grid.predict(X_test)

pr_iro = precision_score(y_test, y_pred, pos_label=1)
pr_no_iro = precision_score(y_test, y_pred, pos_label=0)

rec_iro = recall_score(y_test, y_pred, pos_label=1)
rec_no_iro = recall_score(y_test, y_pred, pos_label=0)

f1_iro = f1_score(y_test, y_pred, pos_label=1)
f1_no_iro = f1_score(y_test, y_pred, pos_label=0)
f1_mean = f1_score(y_test, y_pred, average='macro')

acc = accuracy_score(y_test, y_pred,)

print('Precision iro: {0:.3f}'.format(pr_iro))
print('Precision no iro: {0:.3f}'.format(pr_no_iro))
print('Recall iro: {0:.3f}'.format(rec_iro))
print('Recall no_iro: {0:.3f}'.format(rec_no_iro))
print('F1 iro: {0:.3f}'.format(f1_iro))
print('F1 no_iro: {0:.3f}'.format(f1_no_iro))
print('F1 mean: {0:.3f}'.format(f1_mean))
print('Accuracy: {0:.3f}'.format(acc))

cm = confusion_matrix(y_test, y_pred, labels=grid.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=grid.classes_)
disp.plot()
plt.show()

logger.run['irony/pr_iro'] = pr_iro
logger.run['irony/pr_no_iro'] = pr_no_iro
logger.run['irony/rec_iro'] = rec_iro
logger.run['irony/rec_no_iro'] = rec_no_iro
logger.run['irony/f1_iro'] = f1_iro
logger.run['irony/f1_no_iro'] = f1_no_iro
logger.run['irony/f1_macro'] = f1_mean
logger.run['irony/acc'] = acc

logger.run["irony/confusion_matrix"].upload(neptune.types.File.as_image(disp.figure_))



#-------------------SUBJECTIVITY------------------------
print('-------------------SUBJECTIVITY------------------------')

X_train = train_df['hs'].to_list()
y_train = train_df['subj'].to_list()

X_test = test_df['hs'].to_list()
y_test = test_df['subj'].to_list()


param_grid = {'C': [0.1, 0.5, 1, 10],
              'gamma': ['scale', 1, 0.1, 0.01],
              'degree': [3, 4, 5],
              'kernel': ['rbf','poly']}


grid = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid, refit = True, verbose = 3, scoring=make_scorer(f1_score, pos_label=1), cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

y_pred = grid.predict(X_test)


pr_subj = precision_score(y_test, y_pred, pos_label=1)
pr_obj = precision_score(y_test, y_pred, pos_label=0)

rec_subj = recall_score(y_test, y_pred, pos_label=1)
rec_obj = recall_score(y_test, y_pred, pos_label=0)

f1_subj = f1_score(y_test, y_pred, pos_label=1)
f1_obj = f1_score(y_test, y_pred, pos_label=0)
f1_mean = f1_score(y_test, y_pred, average = 'macro')

acc = accuracy_score(y_test, y_pred,)

print('Precision subj: {0:.3f}'.format(pr_subj))
print('Precision obj: {0:.3f}'.format(pr_obj))
print('Recall subj: {0:.3f}'.format(rec_subj))
print('Recall obj: {0:.3f}'.format(rec_obj))
print('F1 subj: {0:.3f}'.format(f1_subj))
print('F1 obj: {0:.3f}'.format(f1_obj))
print('F1 mean: {0:.3f}'.format(f1_mean))
print('Accuracy: {0:.3f}'.format(acc))

cm = confusion_matrix(y_test, y_pred, labels=grid.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=grid.classes_)
disp.plot()
plt.show()

logger.run['subj/pr_subj'] = pr_subj
logger.run['subj/pr_obj'] = pr_obj
logger.run['subj/rec_subj'] = rec_subj
logger.run['subj/rec_obj'] = rec_obj
logger.run['subj/f1_subj'] = f1_subj
logger.run['subj/f1_obj'] = f1_obj
logger.run['subj/f1_macro'] = f1_mean
logger.run['subj/acc'] = acc

logger.run["subj/confusion_matrix"].upload(neptune.types.File.as_image(disp.figure_))
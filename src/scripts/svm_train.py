import os
import sys

import yaml

sys.path.append(os.path.dirname(sys.path[0]))
from ast import literal_eval

import neptune.new as neptune
import pandas as pd
from loggers.neptune_logger import NeptuneLogger
from matplotlib import pyplot as plt
from models.bert_rep import BertRep
from sklearn import svm
import sklearn.metrics
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from utils.utils import seed_everything

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

try: 
    with open (sys.argv[1] + 'config/svm_train.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)
print('config file loaded!')

seed_everything(config['seed'])

logger = NeptuneLogger()
logger.run['config'] = config

train_df = pd.read_csv(sys.argv[1] + 'data/processed/sentipolc/subj_rep_train.csv', converters={'rep': literal_eval, 'spans': literal_eval})

print('Computing embeddings..')

bert_rep = BertRep()

embeddings_method = getattr(bert_rep, config['embeddings_method'])

train_df['hs'] = train_df['spans'].map(embeddings_method).values.tolist()

test_df = pd.read_csv(sys.argv[1] + 'data/processed/sentipolc/subj_rep_test.csv', converters={'rep': literal_eval, 'spans': literal_eval})
test_df['hs'] = test_df['spans'].map(embeddings_method).values.tolist()

X_train = train_df['hs'].to_list()
X_test = test_df['hs'].to_list()

def train_test(X_train, y_train, X_test, y_test, param_grid, scorer, task):
    print('-------------------' + task + '------------------------')

    grid = GridSearchCV(svm.SVC(class_weight = 'balanced'), param_grid, refit = True, verbose = 3, scoring=scorer, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(grid.best_params_)
    print(grid.best_estimator_)

    y_pred = grid.predict(X_test)

    pr_1 = precision_score(y_test, y_pred, pos_label=1)
    pr_0 = precision_score(y_test, y_pred, pos_label=0)

    rec_1 = recall_score(y_test, y_pred, pos_label=1)
    rec_0 = recall_score(y_test, y_pred, pos_label=0)

    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_mean = f1_score(y_test, y_pred, average='macro')

    acc = accuracy_score(y_test, y_pred)

    print('Precision iro: {0:.3f}'.format(pr_1))
    print('Precision no iro: {0:.3f}'.format(pr_0))
    print('Recall iro: {0:.3f}'.format(rec_1))
    print('Recall no_iro: {0:.3f}'.format(rec_0))
    print('F1 iro: {0:.3f}'.format(f1_1))
    print('F1 no_iro: {0:.3f}'.format(f1_0))
    print('F1 mean: {0:.3f}'.format(f1_mean))
    print('Accuracy: {0:.3f}'.format(acc))

    cm = confusion_matrix(y_test, y_pred, labels=grid.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=grid.classes_)
    disp.plot()
    plt.show()

    logger.run[task+'/pr_1'] = pr_1
    logger.run[task+'/pr_0'] = pr_0
    logger.run[task+'/rec_1'] = rec_1
    logger.run[task+'/rec_0'] = rec_0
    logger.run[task+'/f1_1'] = f1_1
    logger.run[task+'/f1_0'] = f1_0
    logger.run[task+'/f1_macro'] = f1_mean
    logger.run[task+'/acc'] = acc

    logger.run[task+'/confusion_matrix'].upload(neptune.types.File.as_image(disp.figure_))
    logger.run[task+'/best_params'] = grid.best_params_


if 'irony' in config['task']: 
    y_train = train_df['iro'].to_list()   
    y_test = test_df['iro'].to_list()
    train_test(X_train, y_train, X_test, y_test, config['param_grid'], make_scorer(**config['scorer']), 'irony')

if 'subjectivity' in config['task']:
    y_train = train_df['subj'].to_list()   
    y_test = test_df['subj'].to_list()
    train_test(X_train, y_train, X_test, y_test, config['param_grid'], make_scorer(**config['scorer']), 'subjectivity')

if 'polarity' in config['task']:
    y_train = train_df['opos'].to_list()   
    y_test = test_df['opos'].to_list()
    train_test(X_train, y_train, X_test, y_test, config['param_grid'], make_scorer(**config['scorer']), 'polarity/pos')

    y_train = train_df['oneg'].to_list()   
    y_test = test_df['oneg'].to_list()
    train_test(X_train, y_train, X_test, y_test, config['param_grid'], make_scorer(**config['scorer']), 'polarity/neg')


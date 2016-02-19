from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import random

from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import (
                            accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_curve,
                            roc_auc_score,
                            confusion_matrix,
                            classification_report
                            )
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

from unbalanced_dataset import OverSampler, SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


def pickleme(temp, filename):
   with open(filename, 'wb') as picklefile:
       pickle.dump(temp, picklefile)

def getpickle(filename):

    try:
        with open(filename, 'rb') as picklefile:
            return pickle.load(picklefile)
    except:
        print('There was an error trying to read this file.  Please check the filename or path.')

def getScoreValues(X_train,
                  X_test,
                  y_train,
                  y_test,
                  model=KNeighborsClassifier(n_neighbors=6),
                  verbose=True,
                  get_features=True,
                  get_prediction=False
                 ):
    y_test_index=y_test.index

    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_score_value = accuracy_score(y_test, y_pred)
    precision_score_value = precision_score(y_test, y_pred)
    recall_score_value = recall_score(y_test, y_pred)
    f1_score_value = f1_score(y_test, y_pred)
    roc_auc_value = roc_auc_score(y_test,y_pred)

    feature_importances = None
    if get_features:
       try:
           feature_importances = model.feature_importances_
           #print(feature_importances)
       except AttributeError:
           pass

    if verbose:
       #print(pd.concat([y_test,pd.Series(y_pred, index=y_test.index)], axis = 1))
       print('Accuracy: {}\nPrecision: {}\nRecall: {}\nf1: {}\nROC AUC: ()'.format(accuracy_score_value, \
                                                                      precision_score_value, \
                                                                      recall_score_value, \
                                                                      f1_score_value, \
                                                                      roc_auc_value))
    elif get_prediction:
       y_pred=pd.Series(y_pred, index=y_test_index)
       return y_pred

    else:
       return accuracy_score_value, \
              precision_score_value, \
              recall_score_value, \
              f1_score_value, \
              roc_auc_value, \
              feature_importances

def getROCcurve(X_train, X_test, y_train, y_test, model=KNeighborsClassifier(n_neighbors=6)):
    model = model
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    auc = roc_auc_score(y_test, y_scores)
    print('AUC: {}'.format(auc))

    fig,ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve')

    fig.set_size_inches(12, 8, forward=True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='best')

def oversample_data(X_t, y_t, ratio):
    x_columns = X_t.columns

    X_t = X_t.reset_index(drop=True).as_matrix()
    y_t = y_t.reset_index(drop=True).as_matrix()

    smote = OverSampler(ratio=ratio, verbose=False)
    smox, smoy = smote.fit_transform(X_t, y_t)
    X_t = pd.DataFrame(smox, columns=x_columns)
    y_t = pd.Series(smoy)
    return X_t, y_t

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not at Risk', 'At Risk'], rotation=45)
    plt.yticks(tick_marks, ['Not at Risk', 'At Risk'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class getCrossValScoresBySSS(object):

    def __init__(self,
                 local_X,
                 local_y,
                 test_size=0.3,
                 n_iter=50
                ):
        self.model_list = [\
                          KNeighborsClassifier(n_neighbors=1), \
                          SVC(gamma=1, C=10, kernel='rbf'), \
                          SVC(),\
                          LogisticRegression(), \
                          BernoulliNB(), \
                          GaussianNB(), \
                          RandomForestClassifier(n_estimators=30), \
                          DecisionTreeClassifier() \
                          ]

        self.index_func = [\
                          'KNeighborsClassifier(n_neighbors=1)', \
                          'SVC(gamma=1, C=10, kernel=\'rbf\')', \
                          'SVC()',\
                          'LogisticRegression()', \
                          'BernoulliNB()', \
                          'GaussianNB()', \
                          'RandomForestClassifier(n_estimators=30)', \
                          'DecisionTreeClassifier()' \
                          ]
        self.test_size = test_size
        self.n_iter = n_iter
        self.initialize_scores()
        self.score_dict = {}
        self.colors =  getpickle('../security/tableaucolors.pkl')
        self.X = local_X
        self.y = local_y
        self.x_cols = list(local_X.columns)
        self.ratio = float(local_y.value_counts()[0]) / (float(local_y.value_counts()[1]))
        self.sss = StratifiedShuffleSplit(local_y, n_iter=n_iter, test_size=test_size)

    def initialize_scores(self):
        self.acc_array = []
        self.prec_array = []
        self.recall_array = []
        self.f1_array = []
        self.roc_auc_array = []
        self.feature_imp_array = []

    def get_cm_pred(self, model, X_t, y_t, X_ts):
        temp_model = model
        temp_model.fit(X_t, y_t)
        y_pd = temp_model.predict(X_ts)
        return y_pd

    def test_conversion_for_year(self, year, y_delta):
        #try:
            new_year = year+y_delta
            X_columns = self.X.columns
            y_col = self.y.name
            X_te = self.X.loc[new_year, :]
            y_te = self.y.loc[new_year, :]
            y_te = y_te.reset_index('countrycode').reset_index(drop=True).set_index('countrycode')
            full = pd.concat([X_te, y_te], axis=1)
            full = full.dropna()
            X_te = full[X_columns]
            y_te = full[y_col]
            return X_te, y_te
            
        #except:
         #   print('Bad Year')
            
    #def future_prediction_generator(self, )   
    

    def set_score_arrays(self, oversample, year, year_delta, model, get_features):
        random_num = random.randint(1,self.n_iter)
        i = 1
        
        if year != None:
            new_x_cols = self.X.columns
            new_y_col = self.y.name
            new_X = deepcopy(self.X)
            new_y = deepcopy(self.y)
            new_y = new_y.reset_index()
            new_y.year = new_y.year.subtract(year_delta)
            new_y = new_y.set_index(['year','countrycode'])
            new_df = pd.DataFrame()
            new_df = pd.concat([new_X, new_y], axis=1)
            new_df = new_df.dropna()
            new_X = new_df[new_x_cols]
            new_y = new_df[new_y_col]

            new_sss = StratifiedShuffleSplit(new_y, n_iter=self.n_iter, test_size=self.test_size)

            for train_index, test_index in new_sss:
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = new_X.ix[train_index], new_X.ix[test_index]
                y_train, y_test = new_y.ix[train_index], new_y.ix[test_index]

                #remove data of test year from train set
                X_train = X_train.loc[X_train.index.get_level_values('year') < year]
                y_train = y_train.loc[y_train.index.get_level_values('year') < year]

                #Year prior is used later as the comparison between the desired year and its year prior
                _,year_prior = X_te = new_X.loc[year-1, :], new_y.loc[year-1, :]
                X_test, y_test = X_te = new_X.loc[year, :], new_y.loc[year, :]

                if oversample:
                    X_train, y_train = oversample_data(X_train, y_train, self.ratio)
                
                if i == random_num:
                    self.temp_X_train, self.temp_y_train, self.temp_X_test, self.temp_y_test, = X_train, y_train, X_test, y_test

                accuracy_score_value, precision_score_value, \
                recall_score_value, f1_score_value, roc_auc_value, \
                feature_importances = getScoreValues(X_train,
                                                     X_test,
                                                     y_train,
                                                     y_test,
                                                     model=model,
                                                     verbose=False,
                                                     get_features=get_features)

                self.acc_array.append(accuracy_score_value)
                self.prec_array.append(precision_score_value)
                self.recall_array.append(recall_score_value)
                self.f1_array.append(f1_score_value)
                self.roc_auc_array.append(roc_auc_value)
                self.feature_imp_array.append(feature_importances)
                
                i += 1
        
        else:
            for train_index, test_index in self.sss:
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = self.X.ix[train_index], self.X.ix[test_index]
                y_train, y_test = self.y.ix[train_index], self.y.ix[test_index]

                #remove data of test year and all years after from train set
                if year != None:
                    X_train = X_train.loc[X_train.index.get_level_values('year') < year+year_delta]
                    y_train = y_train.loc[y_train.index.get_level_values('year') < year+year_delta]
                    X_test, y_test = self.test_conversion_for_year(year=year, y_delta=year_delta)

                if oversample:
                    X_train, y_train = oversample_data(X_train, y_train, self.ratio)
                
                if i == random_num:
                    self.temp_X_train, self.temp_y_train, self.temp_X_test, self.temp_y_test, = X_train, y_train, X_test, y_test

                accuracy_score_value, precision_score_value, \
                recall_score_value, f1_score_value, roc_auc_value, \
                feature_importances = getScoreValues(X_train,
                                                     X_test,
                                                     y_train,
                                                     y_test,
                                                     model=model,
                                                     verbose=False,
                                                     get_features=get_features)

                self.acc_array.append(accuracy_score_value)
                self.prec_array.append(precision_score_value)
                self.recall_array.append(recall_score_value)
                self.f1_array.append(f1_score_value)
                self.roc_auc_array.append(roc_auc_value)
                self.feature_imp_array.append(feature_importances)
                
                i += 1

    def multi_plot_multi_model_metrics(self):
        index = list(range(len(self.model_list)))
        bw = 0.35
        score_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROCAUC']

        plt.figure(figsize=(18,5))

        for j, scoring in enumerate(score_list):
            ax = plt.subplot(151 + j)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            sns.set_style("whitegrid")

            plt.bar(index, self.score_dict[scoring], bw,
                    align = 'center',
                    #color = colors[(i*2)],
                    alpha = 0.6,
                    label = self.index_func)

            plt.title(scoring, fontsize=15, fontweight='bold')
            plt.xticks(index, self.index_func, rotation='vertical')
            plt.ylim(0.0, 1.1)
            if j == 0:
                plt.ylabel('Score',fontsize=20, fontweight='bold')
            #if j == 4:
            #    plt.legend()
            plt.grid(False)

    def single_plot_multi_model_metrics(self):
        default_index = list(range(len(self.model_list)))
        bw = 0.15
        score_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROCAUC']

        plt.figure(figsize=(18,5))

        for j,scoring in enumerate(score_list):
            ax = plt.subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            sns.set_style("whitegrid")

            index = [i+bw*j for i in default_index]
            plt.bar(index, self.score_dict[scoring], bw,
                    align = 'center',
                    color = self.colors[(3+j)],
                    alpha = 0.6,
                    label = scoring)

            plt.title('Scores for Different Models', fontsize=15, fontweight='bold')
            tick_location = [i for i in default_index]
            plt.xticks(tick_location, self.index_func, rotation=60)
            plt.ylim(0.0, 1.1)
            if j == 0:
                plt.ylabel('Score',fontsize=20, fontweight='bold')
            if j == 4:
                plt.legend(loc='best')
            plt.grid(False)

    def get_multi_models(self,
                         oversample=True,
                         year=None,
                         year_delta=0,
                         make_single_plot=False,
                         make_multi_plots=False
                        ):
        self.score_dict = {}
        models_acc = []
        models_prec = []
        models_rec = []
        models_f1 = []
        models_roc_auc = []

        col_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROCAUC']

        for a_model in self.model_list:
            self.initialize_scores()

            self.set_score_arrays(oversample=oversample,
                                  year=year,
                                  year_delta=year_delta,
                                  model=a_model,
                                  get_features=False)

            mean_acc = np.mean(self.acc_array)
            mean_prec = np.mean(self.prec_array)
            mean_recall = np.mean(self.recall_array)
            mean_f1 = np.mean(self.f1_array)
            mean_roc_auc = np.mean(self.roc_auc_array)

            models_acc.append(mean_acc)
            models_prec.append(mean_prec)
            models_rec.append(mean_recall)
            models_f1.append(mean_f1)
            models_roc_auc.append(mean_roc_auc)

        self.score_dict['Accuracy'] = models_acc
        self.score_dict['Precision'] = models_prec
        self.score_dict['Recall'] = models_rec
        self.score_dict['F1'] = models_f1
        self.score_dict['ROCAUC'] = models_roc_auc

        if make_multi_plots:
            self.plot_multi_model_metrics()
        elif make_single_plot:
            self.single_plot_multi_model_metrics()

        df = pd.DataFrame(self.score_dict, columns=col_names, index=self.index_func)
        return df

    def get_single_model(self,
                         model=KNeighborsClassifier(n_neighbors=2),
                         oversample=True,
                         year=None,
                         year_delta=0,
                         get_features=True,
                         get_cm=False,
                         verbose=True,
                         output_prediction=False,
                         output_filename='',
                         return_features=False
                        ):

        self.initialize_scores()

        self.set_score_arrays(oversample=oversample,
                              year=year,
                              year_delta=year_delta,
                              model=model,
                              get_features=get_features)
        
        if get_cm:
            # For getting a confusion matrix of the last model in the cross validated set
            y_pred = self.get_cm_pred(model, self.temp_X_train, self.temp_y_train, self.temp_X_test)
            
            if output_prediction:
                pred_df = pd.DataFrame(y_pred, index=self.temp_X_test.index)
                pred_df.to_csv(output_filename)
                
            cm = confusion_matrix(self.temp_y_test, y_pred)
            print(cm)
            plt.figure()
            plot_confusion_matrix(cm)

        mean_acc = np.mean(self.acc_array)
        mean_prec = np.mean(self.prec_array)
        mean_recall = np.mean(self.recall_array)
        mean_f1 = np.mean(self.f1_array)
        mean_roc_auc = np.mean(self.roc_auc_array)
        try:
            mean_feature_imp = np.mean(self.feature_imp_array, axis=0)
        except TypeError:
            mean_feature_imp = None

        if verbose:
            print('Avg Feature Importance: {}'.format(mean_feature_imp))
            print('Accuracy: {}\nPrecision: {}\nRecall: {}\nf1: {}\nROC AUC: {}'.format(mean_acc, \
                                                                                        mean_prec, \
                                                                                        mean_recall,\
                                                                                        mean_f1, \
                                                                                        mean_roc_auc))

        if return_features:
            return zip(self.x_cols, mean_feature_imp)
            
        else:
            return mean_acc, mean_prec, mean_recall, mean_f1, mean_roc_auc

    def future_crisis_countries(self,
                                model=KNeighborsClassifier(n_neighbors=2),
                                oversample=True,
                                year=None,
                                year_delta=0):
        assert year != None
        
        new_x_cols = self.X.columns
        new_y_col = self.y.name
        new_X = deepcopy(self.X)
        new_y = deepcopy(self.y)
        #print(len(new_y), len(new_X))
        new_y = new_y.reset_index()
        new_y.year = new_y.year.subtract(year_delta)
        new_y = new_y.set_index(['year','countrycode'])
        new_df = pd.DataFrame()
        new_df = pd.concat([new_X, new_y], axis=1)
        new_df = new_df.dropna()
        new_X = new_df[new_x_cols]
        new_y = new_df[new_y_col]
        #print(len(new_y), len(new_X))
        
        new_sss = StratifiedShuffleSplit(new_y, n_iter=self.n_iter, test_size=self.test_size)

        for train_index, test_index in new_sss:
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = new_X.ix[train_index], new_X.ix[test_index]
            y_train, y_test = new_y.ix[train_index], new_y.ix[test_index]

            #remove data of test year from train set
            X_train = X_train.loc[X_train.index.get_level_values('year') < year]
            y_train = y_train.loc[y_train.index.get_level_values('year') < year]

            #Year prior is used later as the comparison between the desired year and its year prior
            _,year_prior = X_te = new_X.loc[year-1, :], new_y.loc[year-1, :]
            X_test, y_test = X_te = new_X.loc[year, :], new_y.loc[year, :]

            if oversample:
                X_train, y_train = oversample_data(X_train, y_train, self.ratio)

            prediction = getScoreValues(X_train,
                                       X_test,
                                       y_train,
                                       y_test,
                                       model=model,
                                       verbose=False,
                                       get_features=False,
                                       get_prediction=True
                                      )
            prediction_actual = deepcopy(y_test)
            prediction_actual.name = 'prediction_actual'
            year_prior.name='year_prior'
            prediction.name='prediction'
            compare_df = pd.concat([year_prior, prediction, prediction_actual], axis=1)
            compare_df = compare_df.dropna()
            compare_df['net'] = compare_df.year_prior.subtract(compare_df.prediction)
            future_crisis = compare_df[compare_df.net < 0]
            future_crisis = future_crisis[future_crisis.prediction_actual == 1]
            print(future_crisis)

    def SVC_grid_search(self, oversample=True, year=None, year_delta=0):
        # This will perform cross val on one train/test from the SSS
        i = 0
        for tr_i, te_i in self.sss:
            if i == 0:
                train_index = tr_i
                test_index = te_i
            i += 1

        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]

        #remove data of test year and all years after from train set
        if year != None:
            X_train = X_train.loc[X_train.index.get_level_values('year') < year+year_delta]
            y_train = y_train.loc[y_train.index.get_level_values('year') < year+year_delta]
            X_test, y_test = self.test_conversion_for_year(year=year, y_delta=year_delta)

        if oversample:
            X_train, y_train = oversample_data(X_train, y_train, self.ratio)

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'],
                             'gamma': [1,1e-1,1e-2,1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]
                            },
                            {'kernel': ['linear'],
                             'C': [1, 10, 100, 1000]
                            },
                            {'kernel':['poly'],
                             'degree':[1,2,3]
                            }]

        scores = ['accuracy', 'precision', 'recall', 'f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
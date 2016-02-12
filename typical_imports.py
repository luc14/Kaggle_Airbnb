import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics.scorer import log_loss_scorer, accuracy_scorer
from sklearn.model_selection import cross_val_score, train_test_split, ParameterGrid, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
import math, platform, time, warnings, os, traceback, sys, datetime, collections, argparse, io, random, os, re, pprint
#from multiprocessing import Pool
from patsy import dmatrix

import pandas as pd
import numpy as np
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import cross_val_score
import sys
sys.path.append('..')
import common

info_str = '''id: skip
date_account_created: date
timestamp_first_active: date 
date_first_booking: skip
gender: c
age: r 10 80 | b 5 | c
signup_method: c
signup_flow: c
language: c
affiliate_channel: c
affiliate_provider: c
first_affiliate_tracked: c
signup_app: c
first_device_type: c
first_browser: c
country_destination: target
'''    

X, y = common.prepare_data('data/train_users_2.csv', info_str)
#print( X, '\n', y, file = open('out.txt','w') )

logreg = LogisticRegression(random_state=1)
#logreg_score = common.evaluate(logreg, X, y)
logreg.fit(X, y)

prob = logreg.predict_proba(X)
print(prob)

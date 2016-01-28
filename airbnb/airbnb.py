import pandas as pd
import numpy as np
import collections
import sys
sys.path.append('..')
import common
from typical_imports import *

info_str = '''id: id
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

X, y, X_test = common.prepare_data('data/train_users_2.csv','data/test_users.csv', info_str)

logreg = LogisticRegression(random_state=1)
#logreg_score = common.evaluate(logreg, X, y)
logreg.fit(X, y)

prob = logreg.predict_proba(X_test)
print( X, y, X_test, prob, file = open('out.txt','w'), sep= '\n' )


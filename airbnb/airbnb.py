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
def main():
    X, y, X_test = common.prepare_data('data/train_users_2.csv','data/test_users.csv', info_str)
    
    logreg = LogisticRegression(random_state=1)
    #logreg_score = common.evaluate(logreg, X, y)
    logreg.fit(X, y)
    
    
    #print( X, y, X_test, prob, file = open('out.txt','w'), sep= '\n' )
    output_result(logreg, X_test, 'submission.csv')
    
def output_result(learner, X_test, filename):
#def output_result(prob, index_id, countries):
    countries = learner.classes_
    prob = learner.predict_proba(X_test)
    data = pd.DataFrame.from_records(prob, index=X_test.index, columns = countries)
    file = open(filename,'w')
    print('id,country', file = file)
    for user_id in data.index:
        top_countries  = sorted(countries, key = lambda country: data.loc[user_id,country],reverse=True)[:5]
        for i in range(5):
            print(user_id, top_countries[i], file= file, sep=',')
  
if __name__ == '__main__':
    main()  
    
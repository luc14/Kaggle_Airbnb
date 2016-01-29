import pandas as pd
import numpy as np
import collections
import sys
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
    X, y, X_test = common.prepare_data('airbnb/data/train_users_2.csv','airbnb/data/test_users.csv', info_str)

    logreg = LogisticRegression(random_state=1)
    #logreg.fit(X, y)
    
    dummy = DummyClassifier(strategy= 'prior')
    nn = MLPClassifier()
    
    
    #param_dict = {'batch_size': [100, 200], 'momentum': [0.9, 0.99], 'learning_rate_init':[0.001, 0.01, 0.1]}
    #param_dict = {'batch_size': [200], 'momentum': [0.9], 'learning_rate_init':[0.1]}
    #for param in ParameterGrid(param_dict):       
        #nn = MLPClassifier(algorithm='sgd', 
                           #tol=float('-inf'),
                           #warm_start = True,
                           #max_iter=1, 
                           #hidden_layer_sizes = [200])
        #nn_params = nn.get_params()
        #nn_params.update(param)
        #nn.set_params(**nn_params)
        #print(common.evaluate([nn], X, y)) 
        
    print(common.evaluate([dummy, logreg, nn], X, y))#, file = open('evaluation', 'w'))
    #output_result(logreg, X_test, 'submission.csv')



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
    
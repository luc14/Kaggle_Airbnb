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
    print('starting the program: \n\n')
    
    #options = collections.defaultdict(lambda: False)
    options = collections.defaultdict(bool)
    for arg in sys.argv[1:]:
        options[arg] = True
        
        
    if options['small']:
        folder = 'airbnb/data/small_'
    else:
        folder = 'airbnb/data/'
            
    if options['session']:
        sessions = pd.read_csv(folder + 'sessions.csv')
        com_lst = ['action', 'action_type', 'action_detail']
        sessions['new_feature'] = sessions[com_lst].apply(lambda x: tuple(x), axis=1)
        extra_features = common.prepare_counts(sessions, 'new_feature', 'user_id')
    else:
        extra_features = None
        
    X, y, X_test = common.prepare_data(folder + 'train_users_2.csv', folder + 'test_users.csv', extra_features, info_str, options)        
    
    if options['scale']:
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame.from_records(scaler.transform(X), index=X.index, columns= X.columns)    
        X_test = pd.DataFrame.from_records(scaler.transform(X_test), index=X_test.index, columns= X_test.columns)
       
    logreg = LogisticRegression(random_state=1)
    #logreg.fit(X, y)
    
    dummy = DummyClassifier(strategy= 'prior')
    nn = MLPClassifier()
        
    evaluation_metrics = [('acc', accuracy_scorer), ('logloss', log_loss_scorer), ('ndcg', common.ndcg)]
    file = common.create_filename('airbnb')
    print(info_str, file = file)
    print('training data size:', X.shape, file = file)
    print('time:', datetime.datetime.now(), file = file)
    print('arguments:',  sys.argv, file = file)
    all_learners = {
        'dummy':dummy, 
        'logreg': logreg,
        'nn': nn,
    }
    learner_lst = [all_learners[learner] for learner in all_learners if options[learner]]
    new_info = common.evaluate_learners(learner_lst, X, y, evaluation_metrics, options)
    print(new_info, file = file)
    common.combine_info(new_info, file_name = 'summary.txt')
    file.close()
    if options['submission']:
        submission_file = common.create_filename('submission')        
        prepare_submission_file(logreg, X_test, file = submission_file)


def prepare_submission_file(learner, X_test, file):
#def output_result(prob, index_id, countries):
    countries = learner.classes_
    prob = learner.predict_proba(X_test)
    data = pd.DataFrame.from_records(prob, index=X_test.index, columns = countries)
    #file = open(filename,'w')
    print('id,country', file = file)
    for user_id in data.index:
        top_countries  = sorted(countries, key = lambda country: data.loc[user_id,country],reverse=True)[:5]
        for i in range(5):
            print(user_id, top_countries[i], file= file, sep=',')
            



if __name__ == '__main__':
    main()  
    
'''

              DummyClassifier  LogisticRegression  MLPClassifier
accuracymean         0.583473            0.609953       0.531126
accuracystd          0.000022            0.007191       0.038225
log_lossmean        -1.163060           -1.116772      -1.389046
log_lossstd          0.000149            0.019170       0.227440
ndcgmean             0.806765            0.816308       0.763464
ndcgstd              0.000027            0.002626       0.007731

'''
import pandas as pd
import numpy as np
import collections
import sys
import common
from typical_imports import *


'''
age: range, 10, 80; bins, 5; cat

now
info_dict is a dict of lists of lists

info_dict['range'] = [['age', [10,80]], ['anotherage', [20,30]], 'abc': []]

you can make
info_dict is a dict of dict with outside keys = tags

info_dict['range'] = {'age': [10,80], 'anotherage': [20,30], 'abc': []}


or a dict of dicts with outside keys = columns
if you use ordereddict, you can choose the order inside info_str
info_dict = {'age': {'range': [10,80], 'bins': [5], 'cat': []}, 'age1': {'bins': [5], 'range': [10,80], 'cat': []}}


'''

info_str = '''id: id
date_account_created: skip
timestamp_first_active: date, %Y%m%d%H%M%S, dayofyear
date_first_booking: skip
gender: cat
age: range, 10, 80; bins, 5; cat
signup_method: cat
signup_flow: cat
language: cat
affiliate_channel: cat
affiliate_provider: cat
first_affiliate_tracked: cat
signup_app: cat
first_device_type: cat
first_browser: cat
country_destination: target
'''    
def main():
    print('starting the program: \n\n')
    
    #options = collections.defaultdict(lambda: False)
    options = collections.defaultdict(bool)
    for arg in sys.argv[1:]:
        options[arg] = True
        
        
    if options['small']:
        folder = 'airbnb/data/reduced_'
    else:
        folder = 'airbnb/data/'
                    
    info_dict = common.read_info_str(info_str)
    train_data = pd.read_csv(folder + 'train_users_2.csv')
    test_data = pd.read_csv(folder + 'test_users.csv')  
    data = pd.concat([train_data, test_data], ignore_index=True)
    data = common.transform_features(info_dict, data)
    
    if options['session']:
        sessions = pd.read_csv(folder + 'sessions.csv')
        #com_lst = ['action', 'action_type', 'action_detail']
        #sessions['new_feature'] = sessions[com_lst].apply(lambda x: tuple(x), axis=1)
        extra_features = common.prepare_counts(sessions, 'action', 'user_id')
        data = pd.concat([data, extra_features], axis= 1, join = 'inner')
        
    data = shuffle(data, random_state = 1)
    X_test, X, y = common.split_test_train_y(data, target_column='country_destination')
    
    if options['scale']:
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame.from_records(scaler.transform(X), index=X.index, columns= X.columns)    
        X_test = pd.DataFrame.from_records(scaler.transform(X_test), index=X_test.index, columns= X_test.columns)
       
    logreg = LogisticRegression(random_state=1) 
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
    cv = common.split_validation(X, 0.4)
    new_info = common.evaluate_learners(learner_lst, X, y, evaluation_metrics, cv, options)
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
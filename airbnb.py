import pandas as pd
import numpy as np
import collections
import sys
import common
from typical_imports import *

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
    train_data = common.read_file(folder + 'train_users_2.csv', info_dict)
    test_data = common.read_file(folder + 'test_users.csv', info_dict)  
    original_data = pd.concat([train_data, test_data], ignore_index=True)
    data = original_data.copy()
    common.transform_features(info_dict, data)
    
    if options['session']:
        sessions = pd.read_csv(folder + 'sessions.csv')
        extra_features = common.prepare_counts(sessions, 'action', 'user_id')
        data = pd.concat([data, extra_features], axis= 1, join = 'outer')
        
        
    data = shuffle(data, random_state = 1)
    X_test, X, y = common.split_test_train_y(data, target_column='country_destination')
    
    if options['scale']:
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame.from_records(scaler.transform(X), index=X.index, columns= X.columns)    
        X_test = pd.DataFrame.from_records(scaler.transform(X_test), index=X_test.index, columns= X_test.columns)
        
    evaluation_metrics = [('acc', accuracy_scorer), ('logloss', log_loss_scorer), ('ndcg', common.ndcg)]
    file_name = common.create_filename('airbnb')
    file = open(file_name, 'w')
    print(info_str, file = file)
    print('training data size:', X.shape, file = file)
    print('time:', datetime.datetime.now(), file = file)
    print('arguments:',  sys.argv, file = file)

    all_learners = {
        'dummy':DummyClassifier(strategy= 'prior'), 
        'logreg': LogisticRegression(random_state=1),
        'nn': MLPClassifier(),
        'xgb': XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, subsample=0.5, colsample_bytree=0.5,seed=0) 
    }
    learner_lst = [all_learners[learner] for learner in all_learners if options[learner]]
    
    cv = common.split_validation(X, 0.4, condition=lambda row: original_data['timestamp_first_active'].dt.year[row.name] == 2014)
    
    new_info = common.evaluate_learners(learner_lst, X, y, evaluation_metrics, cv, options)    
    changes = open('changes.txt')
    for line in changes:
        new_info[line.strip()] = True
    print(new_info, file = file)
    common.add_info_to_file(new_info, file_name = 'summary.txt')
    file.close()
    if options['submission']:
        for learner in all_learners:
            if options[learner]:
                submission_file = 'submission_'+ learner + '_' + file_name        
                prepare_submission_file(all_learners[learner], X_test, file = open(submission_file, 'w'))


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
    

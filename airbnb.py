import pandas as pd
import numpy as np
import collections
import sys
import common
from typical_imports import *


def main(info_str, args):
    timer = common.Timer()
    
    print('starting the program: \n\n')
    
    #options = collections.defaultdict(lambda: False)
    options = collections.defaultdict(bool)
    for arg in args:
        options[arg] = True
        
        
    if options['small']:
        folder = 'airbnb/data/reduced_'
    else:
        folder = 'airbnb/data/'
                 
    info_dict = common.read_info_str(info_str)
    
    train_data = common.read_file(folder + 'train_users_2.csv', info_dict)
    test_data = common.read_file(folder + 'test_users.csv', info_dict)  
    
    timer.record('reading data')
    
    original_data = pd.concat([train_data, test_data])
    data = original_data.copy()

    common.transform_features(info_dict, data)
    timer.record('transform features')
    
    #timer.restart()
    if options['session']:
        sessions = pd.read_csv(folder + 'sessions.csv')
        extra_features = common.prepare_counts(sessions, 'action', 'user_id')
        data = pd.concat([data, extra_features], axis= 1, join = 'outer')
    timer.record('sessions')
    
    data = shuffle(data, random_state = 1)
    X_test, X, y = common.split_test_train_y(data, target_column='country_destination')
    
 
    #timer.restart()
    if options['scale']:
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame.from_records(scaler.transform(X), index=X.index, columns= X.columns)    
        X_test = pd.DataFrame.from_records(scaler.transform(X_test), index=X_test.index, columns= X_test.columns)
    timer.record('shuffle split and scale')
    
    evaluation_metrics = [('acc', accuracy_scorer), ('logloss', log_loss_scorer), ('ndcg', common.ndcg)]
    
    
    if options['test']:
        output_folder = 'tests/outputs/'
    else:
        output_folder = './'
    file_name = common.create_filename('airbnb', output_folder)
    file = open(output_folder + file_name, 'w')
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
    timer.record('train')

    try:
        changes = open('changes.txt')
        for line in changes:
            line = line.strip()
            if not line:
                continue
            new_info[line] = True
    except:
        pass
    print(new_info, file = file)
    common.add_info_to_file(new_info, file_name = output_folder + 'summary.txt')
    file.close()
    if options['submission']:
        for learner in all_learners:
            if options[learner]:
                all_learners[learner].fit(X, y)
                submission_file = output_folder + 'submission_'+ learner + '_' + file_name        
                prepare_submission_file(all_learners[learner], X_test, file = open(submission_file, 'w'))
                timer.record('submission ' + learner)
                
                
    timer.report()



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
    info_str = open('info_str.txt', 'r').read()
    main(info_str, sys.argv[1:])  
    

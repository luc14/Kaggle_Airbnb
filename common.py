from typical_imports import * 
pd.set_option('display.width', 0)
warnings.filterwarnings('ignore')

def evaluate(learner_lst, X, y):
    evaluation_metrics = [accuracy_scorer, log_loss_scorer, ndcg]
    results = {}
    for learner in learner_lst:
        print(learner, flush=True)
        learner.fit(X, y)        
        #try:
        result = {}
        for metric in evaluation_metrics:
            cv = cross_val_score(learner, X, y, scoring=metric, cv=3)
            result[str(metric) + 'mean'] = cv.mean()
            result[str(metric) + 'std'] = cv.std()
            result[str(metric) + 'training error'] = metric(learner, X, y)
        results[learner.__class__.__name__] = result
        #except Exception as e:
            #print(traceback.format_exc())
            #print(learner, 'failed')
    results_df = pd.DataFrame(results)
    return results_df

def prepare_data(train_filename, test_filename, info_str):
    info_dict = read_info_str(info_str)
    train_data = pd.read_csv(train_filename, parse_dates= info_dict['date'])
    test_data = pd.read_csv(test_filename, parse_dates= info_dict['date'])
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    data.index = data[info_dict['id'][0]]
    for column in info_dict['date']:
        data[column+'_month'] = data[column].dt.month
        info_dict['c'].append(column+'_month')
        data[column+'_year'] = data[column].dt.year
        #data[column+'_dayofyear'] = data[column].dt.dayofyear
        data[column+'_hour'] = data[column].dt.hour // 6 
        info_dict['c'].append(column+'_hour')
    
    for column, arg in info_dict['r']:
        min_, max_ = map(int, arg)
        data.loc[(data[column]<min_) | (data[column]>max_), column] = None
    
    for column, arg in info_dict['b']:
        bins_ = int(arg[0])
        data[column] = pd.cut(data[column], bins_)
        
    data = pd.get_dummies(data,columns=info_dict['c']) 
    
    
    assert len(info_dict['target'])==1
    y = data[info_dict['target'][0]]
    y_train = y[:train_data.shape[0]]
    data.drop(info_dict['skip']+info_dict['date']+info_dict['target']+info_dict['id'], axis=1, inplace=True)
    X_train = data.iloc[:train_data.shape[0]]
    X_test = data.iloc[train_data.shape[0]:]
    
    return X_train, y_train, X_test

def read_info_str(info_str):
    info_dict = collections.defaultdict(list)
    for item in info_str.splitlines():
        column, tags = item.split(':')
        for tag in tags.split('|'):
            tag = tag.strip()
            tag_lst = tag.split()
            if len(tag_lst) > 1:
                tag = tag_lst[0]
                arg = tag_lst[1:]
                info_dict[tag].append([column,arg])
            else:
                info_dict[tag].append(column)
    return info_dict


def ndcg(learner, X, y):
    countries = learner.classes_
    prob = learner.predict_proba(X)
    data = pd.DataFrame.from_records(prob, index=X.index, columns = countries) 
    score = 0
    for user_id in data.index:
        top_countries  = sorted(countries, key = lambda country: data.loc[user_id,country],reverse=True)[:5]   
        for i in range(5):
            if top_countries[i] == y.loc[user_id]:
                score += 1/(math.log2(i+2))
    ave = score/len(y)           
    return ave 
    
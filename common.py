from typical_imports import * 
pd.set_option('display.width', 0)

def evaluate(learner, X, y):
    evaluation_metrics = ['accuracy', 'log_loss']
    result = {i: cross_val_score(learner, X, y, scoring=i, cv=3).mean() for i in evaluation_metrics}
    print(result)
    return result

def prepare_data(filename, info_str):
    info_dict = read_info_str(info_str)
    data = pd.read_csv(filename, parse_dates= info_dict['date'])
    
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
    data.drop(info_dict['skip']+info_dict['date']+info_dict['target'], axis=1, inplace=True)
    X = data
    return X, y

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


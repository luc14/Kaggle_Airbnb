from typical_imports import * 
pd.set_option('display.width', 0)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
warnings.filterwarnings('ignore')

def evaluate(learner, X, y, evaluation_metrics, options):
    if options['parallel']:
        n_jobs = -1
    else:
        n_jobs = 1
    print(learner, flush=True)
    learner.fit(X, y)
    try:
        result = {}
        for name, metric in evaluation_metrics:
            if not options['nocv']:
                cv = cross_val_score(learner, X, y, scoring=metric, cv=3, n_jobs= n_jobs)
                result[name + ' cv mean'] = cv.mean()
                result[name + ' cv std'] = cv.std()
            result[name + ' train'] = metric(learner, X, y)
    except Exception as e:
        print(traceback.format_exc())
        print(learner, 'failed')
    #results_df = pd.DataFrame(results)
    return result #dict

def evaluate_learners(learner_lst, X, y, evaluation_metrics, options):
    records = []
    column_order = collections.defaultdict(lambda: 100)    
    for learner in learner_lst:
        record = evaluate(learner, X, y, evaluation_metrics, options)
        for key in record:
            column_order[key] = 1 
        record.update(options) 
        params = learner.get_params()
        record.update(params)
        record['learner'] = learner.__class__.__name__
        records.append(record)
    column_order['learner'] = 0
    for option in options:
        column_order[option] = 2
    df = pd.DataFrame(records)
    columns = sorted(df.columns, key= lambda x: column_order[x])
    df = df[columns]
    return df

def combine_info(new_info, file_name):
    try:
        summary = pd.read_csv(file_name, index_col= 0, sep='\t')
    except:
        summary = pd.DataFrame()
    data = pd.concat([summary, new_info], ignore_index=True, axis = 0)
    columns = list(new_info.columns) + list(data.columns - new_info.columns)
    data = data[columns] 
    print( data.to_csv(sep='\t'), file = open(file_name, 'w'))
    
def prepare_counts(data, features, key):
    #data = pd.read_csv(filename)
    data.fillna('nan', inplace = True)
    df = pd.crosstab(data[key], data[features])
    return df

#info_dict = {'age': {'range': ['10','80'], 'bins': ['5'], 'cat': []}}

def read_info_str(info_str):
    '''
    return info_dict
    
    >>> d=read_info_str('age: range, 10, 80; bins, 5; cat')
    >>> d['age']['range']
    ['10', '80']
    >>> d['age']['cat'] 
    []
    '''
    
    info_dict = collections.OrderedDict()
    for item in info_str.splitlines(): # item = 'age: range, 10, 80; bins, 5' / item = 'gender: cat'
        column, tags = item.split(':') # column = 'age' ,    tags = ' range, 10, 80; bins, 5' / column = 'gender', tags = 'cat'
        info_dict[column] =  collections.OrderedDict()
        for tag in tags.split(';'): # tag = ' range, 10, 80' /  
            tag = tag.strip() # get rid of white space (new line charactert, tab, space)
            tag_lst = [ tag_arg.strip() for tag_arg in tag.split(',')] # tag_lst = ['range', '10', '80'] / tag_lst = ['cat']
            tag = tag_lst[0] # tag = 'range'  
            arg = tag_lst[1:] # arg = ['10', '80'] 
            # info_dict['age'] = {'range': ['10', '80'], 'bins': ['5']} 
            info_dict[column][tag] = arg
            
    return info_dict

def transform_features(info_dict, data):
    
    # separate information from date format into either 'c' or continuous variables 
    for column in info_dict['date']:
        data[column+'_month'] = data[column].dt.month
        info_dict['c'].append(column+'_month')
        if options['year']:
            data[column+'_year'] = data[column].dt.year
        #data[column+'_dayofyear'] = data[column].dt.dayofyear
        data[column+'_hour'] = data[column].dt.hour // 6 
        info_dict['c'].append(column+'_hour')
    
    # cut data into the range of [min_, max_]
    for column, arg in info_dict['r']:
        min_, max_ = map(int, arg)
        data.loc[(data[column]<min_) | (data[column]>max_), column] = None
    
    #cut data into (bins_)'s groups
    for column, arg in info_dict['b']:
        bins_ = int(arg[0])
        data[column] = pd.cut(data[column], bins_)
    
    #transform data in 'c' into categorical formats
    data = pd.get_dummies(data,columns=info_dict['c']) 
    
    
    return tranformed_data

def prepare_data(train_data, test_data, extra_features , info_dict, options):
 
    # combine test data and train data
    data = pd.concat([train_data, test_data], ignore_index=True)
    data.index = data[info_dict['id'][0]]
    
    # combine extra features together
    if extra_features is not None:
        data = pd.concat([data, extra_features], axis= 1, join= 'inner')
        
    if options['shuffle']:
        data = shuffle(data, random_state = 1)
    else:
        data = data.sort_values('timestamp_first_active', axis = 0)
            
    
    #store target values in y_train
    assert len(info_dict['target'])==1
    y = data[info_dict['target'][0]]
    y_train = y[~y.isnull()]
    
    #store features into X_train and X_test
    data.drop(info_dict['skip']+info_dict['target']+info_dict['id']+ info_dict['date'], axis=1, inplace=True)
    X_train = data[~y.isnull()]
    X_test = data[y.isnull()]
    
    return X_train, y_train, X_test


def ndcg(learner, X, y):
    countries = learner.classes_
    prob = learner.predict_proba(X)
    data = pd.DataFrame(prob, index=X.index, columns = countries) 
    score = 0
    for user_id in data.index:
        top_countries  = sorted(countries, key = lambda country: data.loc[user_id,country],reverse=True)[:5]   
        for i in range(5):
            if top_countries[i] == y.loc[user_id]:
                score += 1/(math.log2(i+2))
    ave = score/len(y)           
    return ave 

    
def create_filename(prefix):    
    now = datetime.datetime.now()
    file_date = now.strftime('%Y%m%d%H%M%S')
    lst_names = os.listdir()
    max_num = 0
    for name in lst_names:
        name_digit = re.search('^' +prefix + r'(\d+)_.*\.txt$', name)
        if name_digit is None:
            continue
        digit = int(name_digit.group(1))
        if digit > max_num:
            max_num = digit
            
    file_name = prefix + str(max_num + 1)+ '_' + file_date + '.txt'
    file = open(file_name, 'w')
    return file

def git_version():
    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse','HEAD'], stdout = PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()

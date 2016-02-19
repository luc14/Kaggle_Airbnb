from typical_imports import * 
pd.set_option('display.width', 0)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
warnings.filterwarnings('ignore')

def split_validation(X, fraction, condition = lambda row: True, random_state = 1):
    idx = X.apply(condition, axis = 1)
    idx.index = range(len(X)) # the same as idx.reset_index(drop = True)
    val_index = pd.DataFrame(idx[idx]).sample(frac = fraction, random_state = random_state).index
    train_index = idx.index.difference(val_index)
    return [(train_index, val_index)]

def evaluate(learner, X, y, evaluation_metrics, cv_split, options):
    print(learner, flush=True)
    learner.fit(X, y)
#    try:
    result = {}
    for name, metric in evaluation_metrics:
        score = cross_val_score(learner, X, y, scoring=metric, cv=cv_split)
        result[name + ' cv mean'] = score.mean()
        result[name + ' cv std'] = score.std()
        result[name + ' train'] = metric(learner, X, y)
#except Exception as e:
        #print(traceback.format_exc())
        #print(learner, 'failed')
    #results_df = pd.DataFrame(results)
    return result #dict

def evaluate_learners(learner_lst, X, y, evaluation_metrics, cv, options):
    records = []
    column_order = collections.defaultdict(lambda: 100)    
    for learner in learner_lst:
        record = evaluate(learner, X, y, evaluation_metrics, cv, options)
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

def add_info_to_file(new_info, file_name):
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

def read_info_str(info_str):
    '''
    return info_dict
    
    >>> d=read_info_str('age: range, 10, 80; bins, 5; cat')
    >>> d['age']['range']
    ['10', '80']
    >>> d['age']['cat'] 
    []
    >>> for column in d:
    ...   for tag, arg_lst in d[column].items():
    ...      print(column, tag, arg_lst)
    age range ['10', '80']
    age bins ['5']
    age cat []
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

#info_dict = {'age': {'range': ['10','80'], 'bins': ['5'], 'cat': []}, }


# reads index and dates correctly
def read_file(filename, info_dict):
    data = pd.read_csv(filename)
    for column in info_dict:
        for tag, arg_lst in info_dict[column].items():
            if tag == 'id':
                data.index = data[column]
                data.drop(column, axis=1, inplace=True)
            if tag == 'date':
                data[column] = pd.to_datetime(data[column], format = arg_lst[0])
    return data
                
# will change the input data
def transform_features(info_dict, data):
    #data = data.copy()
    for column in info_dict:
        for tag, arg_lst in info_dict[column].items():            
            if tag == 'date':
                for arg in arg_lst[1:]:
                    if arg == 'trend':
                        #convert date in timestamp into integer
                        data[column + '_' + arg] = data[column].astype(np.int64) 
                        continue
                    data[column + '_' + arg] = getattr(data[column].dt, arg)
                data.drop(column, axis=1, inplace=True)
            
                
            # cut data into the range of [min_, max_]
            if tag == 'range':
                min_, max_ = map(int, arg_lst)
                data.loc[(data[column]<min_) | (data[column]>max_), column] = None
            
            #cut data into (bins_)'s groups
            if tag == 'bins':
                bins_ = int(arg_lst[0])
                data[column] = pd.cut(data[column], bins_, labels= False) 
            
            #convert data in 'c' into categorical formats     
            if tag == 'cat':
                data1 = pd.get_dummies(data, columns = [column], dummy_na = True)
                data.drop(data.columns, inplace=True, axis=1)
                data[data1.columns] = data1
                
            if tag == 'skip':
                data.drop(column, axis=1, inplace=True)
        

def split_test_train_y(data, target_column):
    # info_dict = {'country_destination': {'target': }}
    y = data[target_column]
    test_index = y.isnull()
    data = data.drop(target_column, axis = 1)
    test_X = data.loc[test_index]
    train_X = data.loc[~test_index]
    train_y = y[~test_index]
    #validation_index = train_data.apply(condition, axis = 1)
    #validation_data = train_data.loc[validation_index]
    #validation_data = validation_data.sample(frac = fraction)
    
    return test_X, train_X, train_y

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
    return file_name

def git_version():
    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse','HEAD'], stdout = PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()

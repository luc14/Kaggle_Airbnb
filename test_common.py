import pandas.util.testing as pdt
import pandas as pd
import common
import collections

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
def test_transform_features():
    data = pd.read_csv('tests/reduced_train_users_2.csv')
    output = common.transform_features(common.read_info_str(info_str), data)
    #print (result, file = open('test1', 'w'))    
    expected_output = pd.read_csv('tests/reduced_train_users_2.csv.output', index_col= 'id', sep= '\t', parse_dates= ['timestamp_first_active'])
    pdt.assert_frame_equal(output, expected_output)
  
def test_infostr():
    output = str(common.read_info_str(info_str))
    expected_output = open('tests/info_dict', 'r')
    expected_output = next(expected_output).strip()
    assert output == expected_output

info_str1 = '''timestamp_first_active: date, %Y%m%d%H%M%S, dayofyear
date_first_booking: skip
gender: cat
age: range, 10, 80; bins, 5; cat
'''

def test_infostr2(capsys):
    output = common.read_info_str(info_str1)
    for column in output:
        for tag, arg_lst in output[column].items():
            print(column, tag, arg_lst) 
    out, err = capsys.readouterr()
    assert out == '''timestamp_first_active date ['%Y%m%d%H%M%S', 'dayofyear']
date_first_booking skip []
gender cat []
age range ['10', '80']
age bins ['5']
age cat []
'''
    
def test_infostr1():
    output = common.read_info_str(info_str1)
    expected_output = { 'timestamp_first_active': {'date': ['%Y%m%d%H%M%S', 'dayofyear']},
                        'date_first_booking': {'skip': []},
                        'gender': {'cat': []},
                        'age': {'range': ['10', '80'], 'bins': ['5'], 'cat': []}
                        }
    
    assert output == expected_output
    assert list(output) == ['timestamp_first_active', 'date_first_booking', 'gender', 'age']
    assert list(output['age']) == ['range', 'bins', 'cat']
    

def generate_test_files():
    data = pd.read_csv('airbnb/data/reduced_train_users_2.csv')
    print(common.read_info_str(info_str), file = open('tests/info_dict1', 'w'))
    #result = common.transform_features(common.read_info_str(info_str), data)
    ##print(result.to_csv(sep='\t'), file = open('test1', 'w'))
    #result.to_csv('test_transform_features_1', sep='\t')

    
def condition(row):
    return row['year'] == 2014


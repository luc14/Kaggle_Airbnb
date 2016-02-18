import pandas.util.testing as pdt
import pandas as pd
import common
import collections

info_str = '''id: id
date_account_created: skip
timestamp_first_active: date, %Y%m%d%H%M%S, dayofyear, trend, dayofweek
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
    train_filename = 'tests/reduced_train_users_2.csv'
    info_dict = common.read_info_str(info_str)
    data = common.read_file(train_filename, info_dict) 
    common.transform_features(info_dict, data)
    #print (result, file = open('test1', 'w'))    
    expected_output = pd.read_csv(train_filename + '.expected', index_col= 'id', sep= '\t')
    data.to_csv(train_filename + '.actual', sep='\t')
    # can compare actual to expected files, but then have to deal with a few technical details
    # especially annoying is that float numbers may differ in the last decimal digit
    # assert_frame_equal automatically recognizes that 0.3 and 0.29999999999 are the same
    # so better to use pdt.assert_* instead of comparing files on disk
    pdt.assert_frame_equal(data, expected_output)
  
def test_infostr():
    output = str(common.read_info_str(info_str))
    print(output, file=open('tests/info_dict.actual', 'w'))
    expected_output = open('tests/info_dict.expected', 'r')
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
    

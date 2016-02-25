import pandas.util.testing as pdt
import pandas as pd
import common
import collections
import airbnb
import os 
import glob

def test_airbnb():
    # clean the outputs folder first
    output_folder = 'tests/outputs/'
    expected_output_folder = 'tests/expected_outputs/'
    files = glob.glob(output_folder + '/*')
    for file in files:
        os.remove(file)
    try:
        # if there is no such a folder there, then create a new folder
        os.mkdir(output_folder)
    except:
        pass
        
    info_str = open('tests/info_str.txt', 'r').read()
    airbnb.main(info_str, ['--learners', 'xgb', '--session', '--submission', '--input_folder', 'tests/', '--output_folder', 'tests/outputs/'])
    expected_output_files = sorted(os.listdir(expected_output_folder))
    output_files = sorted(os.listdir(output_folder))
    for output_file, expected_output_file in zip(output_files, expected_output_files):
        outputs = open(output_folder + output_file)
        expected_outputs = open(expected_output_folder + expected_output_file)
        for output_line, expected_output_line in zip(outputs, expected_outputs):
            if output_line.startswith('time:'):
                continue
            assert output_line == expected_output_line
            
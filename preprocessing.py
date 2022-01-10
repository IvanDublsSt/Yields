import gc
import os
import functools
import tqdm
import shutil
import requests
import pandas as pd
import numpy as np

from pyunpack import Archive
from bs4 import BeautifulSoup
from pathlib import Path
from multiprocessing import Pool
from dbfread import DBF
from typing import Union

import re


def load_and_unpack(url:str,
                    file_name:str, 
                    load_path:str, 
                    save_path:str, 
                    overwrite:bool=True) -> None:
    '''
    Loads and unpacks archive from given url
    
    Parameters
    ----------
    url: str
            Url to load file from
    file_name: str
            File name to save file
    load_path: str
            Folder to save rar or zip archives from CBR site
    save_path: str
            Folder to save unpacked .dbf files
    override_data: bool
            Whether to overwrite data in folder if it already exists
    '''

    out = requests.get(url, stream=True)
    archive_path = Path(load_path) / file_name
  
    # сохраним архив
    with open(archive_path, 'wb') as f:
        f.write(out.content)

    # распакуем архив с .dbf-файлами в папку
    unzip_path = Path(save_path) / file_name
    # создадим папку для сохранения распакованного архива
    if Path.exists(unzip_path) and not overwrite:
        print('The folder already exists, no data will be added to existing files')
    else:
        Path.mkdir(unzip_path)
        # распакуем
        Archive(archive_path).extractall(unzip_path)


def load_bank_statements(form_number:int, 
                         filepath: str,
                         overwrite:bool=True) -> None:
    '''
    Loads archives with bank statements from CBR website and unpacks them into given folder
    
    Parameters
    ----------
    form_number: int, 101 or 102
            Number of CBR form of financial statements (form 101, form 102)
    filepath: str
            Directory (folder) to save .zip and .rar archives  downloaded from CBR site
            as well as unpacked .dbf files from archives
    overwrite: bool
            Whether to overwrite directories with archives and unzipped data if they already
            exist (option overwrite=False is currently not available)
    '''

    print('Downloading and unpacking files from www.cbr.ru, please be patient...')
    
    url = 'https://cbr.ru/banking_sector/otchetnost-kreditnykh-organizaciy/'
    
    # create directories to save data
    load_path = Path(filepath) / (str(form_number) +'_zipped')
    save_path = Path(filepath) / str(form_number)
    # delete directories if they already exist
    if overwrite:
        if load_path.is_dir():
            shutil.rmtree(load_path, ignore_errors=True)
        if save_path.is_dir():
            shutil.rmtree(save_path, ignore_errors=True)
            
    # create new empty folders for data instead of the old folders
    Path(load_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html')

    # выберем все ссылки для нужной формы отчётности
    all_refs = [x['href'] for x in soup.find_all('a', href=True)]
    refs = ['https://cbr.ru/' + x for x in all_refs \
            if 'forms/' + str(form_number) in x]
    
    # сгенерируем кортежи аргументов для параллельной функции
    args = ((x, 
             (
             (x.split('/')[-1]).split('.')[0]
             ).split('-')[1], 
             load_path, 
             save_path,
             True) for x in refs)
    
    # параллельные действия
    # if __name__=='__main__':
    with Pool() as pool:
        pool.starmap(load_and_unpack, 
                     # to show progress bar
                     tqdm.tqdm(args, total=len(refs)))
            
    print('Congratulations! Finished.')
            
    # на всякий случай сохранил альтернативный вариант с циклом
    # он работает в 4 раза медленнее, чем распараллеленный вариант
    
    #for ref in refs:
    #    load_and_unpack(ref,
    #                    file_name=(ref.split('/')[-1]).split('.')[0],
    #                    load_path=load_path,
    #                    save_path=save_path)


def dbf2df(filepath: str, encoding:str) -> pd.DataFrame:
    """
    Reads .dbf from given filepath into dataframe and returns df
    """
    dbf = DBF(filepath, encoding=encoding)
    df = pd.DataFrame(iter(dbf)) 
    # if there is no 'DT' column with date
    if 'DT' not in df.columns:
        # get date from folder name
        try:
            date = str(filepath).split('\\')[-2]
        # path with ordinary slash if working on Linux 
        except IndexError:
            date = str(filepath).split('/')[-2]
        df['DT'] = pd.to_datetime(date)
    return df
    

def get_filepaths(filepath:str):
    """Returns full paths to files in the given folder with form 101 or 102"""
    folders = [filelist[0] for filelist in os.walk(filepath)][1:]
    files = []
    for folder in folders:
        for file in os.listdir(folder):
            files.append(Path(folder)/file)
    return files


def get_bank_names(filepath:str, 
                  form_number:int, 
                  encoding:str='cp866') -> pd.DataFrame:
    """
    Collects all bank names and register numbers from given form 
    (from files with 'N1' for form 101 and 'NP1' for form 102). Returns 
    Pandas DataFrame with 2 columns: register number (REGN) and name 
    of the bank.
    
    Parameters
    ----------
    path: str
            Path to folder with folders with downloaded and unzipped files
    encoding:str, default 'cp866'
            Encoding to open .dbf files from CBR. Recommended value is 'cp866'
    form_number: int, 101 or 102
            Whether to collect bank names from files with 101 or 102 form
    """
    # collect all file paths from folders in given path
    files = get_filepaths(filepath)
            
    # different files with names for different forms
    search = {101:'N1', 102:'NP1'}
    # find all files with matching pattern in file names
    names = list(filter(lambda x: \
                        search[form_number] in str(x), files))
    
    # read all these files and merge them into one file
    df = functools.reduce(lambda a,b: \
                          pd.concat((a.reset_index(drop=True), 
                                     b.reset_index(drop=True))),
                          [dbf2df(x, encoding=encoding) for x in names])
    
    # integer codes ('REGN') are unique for all banks, but 
    # the same bank names are sometimes written in different ways
    df.drop_duplicates(subset='REGN', inplace=True)
    return df[['REGN', 'NAME_B']].reset_index(drop=True)


def read_form(filepath:str, 
              form_number:int, 
              which_files:str=None, 
              remove_unknown_accs:bool=True,
              to_int:bool=True, 
              encoding:str='cp866') -> pd.DataFrame:
    '''
    Reads and merges all .dbf files for given form and filepath. Returns merded
    dataframe.

    Parameters
    ----------
    filepath: str
            Directory (folder) where are stored .dbf files for form 101 or 102
    form_number: int, 101 or 102 
            Number of CBR form of financial statements (form 101, form 102 etc)
    which_files: str, default None
            Search pattern to look for in file names. For example, by default the 
            function opens and merges all files with 'B1' in filename for 
            form  101 and all files with '_P1' in filenames for 102 form. You can set your 
            own search pattern, but you should be sure, that all files with that pattern
            have the same column names and column order, otherwise function will return
            garbage.
    remove_unknown_accs: bool, default True
            Whether to remove unknown accounts from columns with account number. 
            There are some accounts in form 101 whose meaning I could not find 
            in the Central Bank documents. These accounts are 'ITGAP', '304.1', '408.1', 
            '408.2', '474.1', '1XXXX', '2XXXX', '3XXXX', '4XXXX', '5XXXX', '6XXXX',
            '7XXXX', '8XXXX', '9XXXX'. Removing them allows us to convert column with
            account number from string to integer. This conversion boosts performance
            in dataframe processing and memory management.
    to_int: bool, default True
            Whether to convert column with account numbers to int.
    enconding: str, default 'cp866'
            Encoding to open .dbf files. 'cp866' works well with form 101 and 102
    '''
    
    print('Reading .dbf files from your PC, please wait...')

    files = get_filepaths(filepath)
    # different files with names for different forms
    search = {101:'B1', 102:'_P1'}
    # columns with account numbers
    acc_cols = {101:'NUM_SC', 102:'CODE'}
    # accounts to remove if remove_unknown_accs=True
    remove_accounts = ['ITGAP', '304.1', '408.1', '408.2', '474.1'] + \
                      [str(x)+'XXXX' for x in range(1, 9, 1)]
    # find all files with matching pattern in file names
    if which_files:
        search_str = which_files
    else:
        search_str = search[form_number]

    names = list(filter(lambda x: search_str in str(x), files))
    
    args = ((name, encoding) for name in names)
    
    with Pool() as pool:
        dfs = list(pool.starmap(dbf2df, 
                     # to show progress bar
                     tqdm.tqdm(args, total=len(names))))

    print('Opened files. Merging them...')

    df = pd.concat(dfs)
    # delete large list of files from memory
    del dfs
    gc.collect()

    # make date column to datetime index
    df.index = pd.to_datetime(df['DT'])
    df.sort_index(inplace=True)
    df.drop(columns='DT', inplace=True)
    # remove unknown accounts and convert account numbers to int
    if all([remove_unknown_accs, form_number==101]):
        df = df[~df['NUM_SC'].isin(remove_accounts)]
    # convert account numbers column to integer
    if to_int:
        if not remove_unknown_accs and form_number==101:
            raise TypeError(
                """
                You have not removed some very specific accounts
                (remove_unknown_accs=False). This accounts (for 
                instance, 3XXXX) can not be converted to integer.
                """
                )
        df[acc_cols[form_number]] = df[acc_cols[form_number]].astype('int32')

    print('Done.')
    return df


def group(data:pd.DataFrame, 
          aggschema:Union[dict, pd.DataFrame], 
          form:int, 
          acc_col:str=None,
          agg_col:str=None,
          reg_col:str='REGN',
          date_col:str='DT', 
          aggfunc:str='sum') -> pd.DataFrame:
    """
    Returns dataframe with accounts values grouped and sumed by 
    unique bank register number, date and aggschema supplied by user
    
    Parameters:
    -----------
    data:pd.DataFrame
        Dataframe to group. Contains at least 4 columns with:
            - bank register numbers
            - dates
            - old account codes (integer datatype)
            - values for old account codes
    aggschema:dict or Pandas DataFrame
        Dictionary of DataFrame wich maps accounts in the data to 
        the grouped accounts for analytical purposes. 
        Example for dict:
            aggschema = {'Retail credits': [45502, 45508, 45509]},
            where 'Retail credits' is the new account name, and 
            45502, 45508, 45509 are old account numbers to be grouped
            into one 'Retail credit' account for each bank in the table.
        Example for DataFrame:
                   new_code    old_code
            0  'Retail credit'  45502
            1  'Retail credit'  45508
            2  'Retail credit'  45509
        The DataFrame should contain new account in the first column and 
        old accounts in the second. The names of the columns are not important.
    form:int, 101 or 102
        CBR form number.
    acc_col:str, default None
        Use specific account column name instead of 'NUM_SC' for form 101
        or 'CODE' for form 102.
    agg_col:str, default None
        Use specific aggregation column name (with numbers to aggregate)
        instead of 'SIM_ITOGO' for form 101 or 'IITG' for form 102.
    date_col:str, defaul 'DT'
        Date column name in dataframe with data. Default 'DT' (both 
        in form 101 and 102)
    aggfunc:str, default 'sum'
        Function to aggregate existing accounts into new accounts.
    """
    account_cols={101:'NUM_SC', 102:'CODE'}
    aggcols = {101:'IITG', 102:'SIM_ITOGO'}
    # if custom account or aggregation columns are submitted
    if acc_col:
        account_cols[form] = acc_col
    elif agg_col:
        aggcols[form] = agg_col

    if isinstance(aggschema, dict):
        newdict = {y:x[0] for x in aggschema.items() for y in x[1]}
        aggschema = pd.DataFrame({'new_code': list(newdict.values()),
                                  account_cols[form]: list(newdict.keys())})
    elif isinstance(aggschema, pd.DataFrame):
        aggschema.columns=['new_code', account_cols[form]]

    print('Grouping and aggregating data. Please be patient...')
    df = pd.merge(left=data.reset_index(), 
                  right=aggschema, 
                  how='left', 
                  left_on=account_cols[form], 
                  right_on=account_cols[form])
    
    df.set_index(date_col, inplace=True)
    df.dropna(subset=[account_cols[form]], inplace=True)
    df = df.groupby(by=[reg_col, 
                        df.index, 
                        'new_code']).agg({aggcols[form]:aggfunc})
    
    df.reset_index(inplace=True)
    df.set_index(date_col, inplace=True)

    print('Finished.')
    
    return df

def preprocess_df(data : pd.DataFrame, form : str) -> pd.DataFrame:
    """
    Returns dataframe with columns for code, level, name and sign (P or A) of the entry
    
    Parameters:
    -----------
    data:pd.DataFrame
        DataFrame to process. Must be one of the pages from Деревья
    """
    if form == "BS":
        data = data[data.columns[0:4]]
    elif form == "PNL":
        data = data[data.columns[[1,2,4,6]]]
    else:
        print(f"There is no such form as {form} implemented")
    data.columns = ["code", "level", "name", "sign"]
    data = data[~data.name.isnull()]
    return data

def create_level_separating_indices(data : pd.DataFrame, level : int) -> np.array:
    """
    Returns array of indices of level borders
    
    Parameters:
    -----------
    data:pd.DataFrame
        Prerocessed DataFrame from preprocess_df function
    
    level:int
        Level of aggregation 
    """
    separating_indices = (data[data.level == level]).index.values
    separating_indices = np.append(separating_indices, data.index.values[-1])
    return separating_indices

def create_level_names(data : pd.DataFrame, level : int) -> np.array:
    """
    Returns array of names of groups
    
    Parameters:
    -----------
    data:pd.DataFrame
        Prerocessed DataFrame from preprocess_df function
    
    level:int
        Level of aggregation 
    """
    separating_indices = (data[data.level == level]).index.values
    group_names = list(data.name[separating_indices])
    return group_names

def limit_df_by_level(data : pd.DataFrame, form : str) -> pd.DataFrame:
    """
    Returns DataFrame with only the lowest-level observations (4 for BS, 8 for PNL)
    
    Parameters:
    -----------
    data:pd.DataFrame
        Prerocessed DataFrame from preprocess_df function
    
    form:str
        Type of the form. "BS" for balance sheet, "PNL" for profit and loss
    """
    if form == "BS":
        lowest_level = 4
    elif form == "PNL":
        lowest_level = 8
        
    data_low_level = data[(data.level == lowest_level)|
                          (data.level.isnull())]
    
    data_low_level = data_low_level[data_low_level.code.isnull() == False]
    
    return data_low_level

def create_tuples_from_separating_indices(separating_indices : list) -> list:
    """
    Returns list tuples with beginning and ending indices of each group
    
    Parameters:
    -----------
    separating_indices:list
        list of indices of the group separators 

    """
    separating_indices_shifted = np.roll(separating_indices, 1)
    separating_tuples = list(zip(separating_indices_shifted, separating_indices))
    separating_tuples.remove(separating_tuples[0])
    return separating_tuples

def zip_all_names_and_boundaries(group_names : list, separating_tuples : list) -> dict:
    """
    Returns dictionary of format {group name: (starting index, ending index)}
    
    Parameters:
    -----------
    group_names:list
        list of indices of the group names
    separating_tuples:list
        list of tuples with beginning and ending values 

    """
    all_names_and_boundaries = dict(zip(group_names, separating_tuples))
    return all_names_and_boundaries

def create_group_dict(data : pd.DataFrame, name : str, all_names_and_boundaries : dict) -> dict:
    """
    Returns dictionary of format {group name: list of codes} for a particular name.
    
    Parameters:
    -----------
    data:pd.DataFrame
        Prerocessed DataFrame from preprocess_df function
    name:str
        Name of the group
    all_names_and_boundaries:dict
        Dictionary of format {group name: (starting index, ending index)}
    """
    separating_mask = data.index.to_series().between(*all_names_and_boundaries[name])
    trees_balance_old_between = data[separating_mask]
    group_dict = {name : list(trees_balance_old_between.code)}
    return group_dict

def create_dictionary_from_name_and_level(data : pd.DataFrame, name : str, level : int, form : str) -> dict:
    """
    Returns dictionary of format {group name: list of codes} for a particular name, level, form from the raw data.
    
    Parameters:
    -----------
    data:pd.DataFrame
        Raw data from Деревья
    name:str
        Name of the group
    level:int
        Level of aggregation 
    form:str
        Type of the form. "BS" for balance sheet, "PNL" for profit and loss
    """
    data = preprocess_df(data, form)
    separating_indices = create_level_separating_indices(data, level)
    group_names = create_level_names(data, level)
    data_low_level = limit_df_by_level(data, form)
    separating_tuples = create_tuples_from_separating_indices(separating_indices)
    all_names_and_boundaries = zip_all_names_and_boundaries(group_names, separating_tuples)
    group_dictionary = create_group_dict(data_low_level, name, all_names_and_boundaries)
    group_dictionary_sorted = sort_one_dictionary(group_dictionary, data_low_level)
    return group_dictionary_sorted

def sort_one_dictionary(dictionary : dict, data : pd.DataFrame) -> dict:
    """
    Returns dictionary of format {group name : {"A" : [codes], "P" : [codes]}}
    
    Parameters:
    -----------
    dictionary:dict
        dictionary of format {group name: list of codes}
    data:pd.DataFrame
        Lower-level data
    """
    
    name = list(dictionary.keys())[0]
    codes = pd.Series(np.array(list(dictionary.values()))[0])

    end_index = check_location(index_values = codes, data = data)
    if end_index == "EmptyIndex":
        empty_dictionary = {name : {"A" : [],
                                    "P" : []}}
        return empty_dictionary
    checker_vectorized = np.vectorize(check_code_is_positive, excluded = ["data", "end_index"])

    data.code = data.code.apply(str)
    positive_mask = pd.Series(
                                checker_vectorized(code = codes, data = data, end_index = end_index)
                                                                                                )
    negative_mask = ~positive_mask
    
    positive_codes = codes[positive_mask]
    negative_codes = codes[negative_mask]
    
    new_dictionary = {name : {"A" : list(positive_codes),
                              "P" : list(negative_codes)}}
    return(new_dictionary)


def check_code_is_positive(code : str, end_index : int,  data : pd.DataFrame) -> bool:
    """
    Returns True if one code is positive and False if it is negative
    
    Parameters:
    -----------

    code:str
        A code
    end_index:int
        Ending index of the slice in the general dataframe, which corresponds to the dictionary
    data:pd.DataFrame
        Preprocessed data from preprocess_df
    """
    code = str(code)
    if end_index == "EmptyIndex":
        return "EmptyIndex"
    if len(data[data.code == code].sign) == 0:
        return False
    elif len(data[data.code == code].sign) > 1:
        index_values = data[data.code == code].index
        index_ranges = index_values - end_index
        index_ranges = index_ranges[index_ranges <= 0]
        index_closest_range = np.max(index_ranges)
        index_needed = index_closest_range + end_index
        code_type = data[data.index == index_needed].sign.item()
    else:
        code_type = data[data.code == code].sign.item()
        

    if code_type == "A":
        return True
    elif code_type == "P":
        return False
    
def check_location(index_values : np.array,  data : pd.DataFrame) -> int:
    """
    Takes an array of integers (the values of dictionary). 
    Locates them in the file, finds ending index, returns it as an integer. 
    
    Parameters:
    -----------
    index_values:np.array
        An array of integers for indices of a code
    data:pd.DataFrame
        Preprocessed data from preprocess_df
    """
    if len(index_values) == 0:
        return "EmptyIndex"
    index_values = pd.Series(index_values).apply(str)

    values_in_values = data.code.apply(str).isin(index_values)
    count_consequtive_values = values_in_values * (values_in_values.groupby((values_in_values !=
                                                           values_in_values.shift()).cumsum()).cumcount() + 1)
    length_of_dictionary = len(index_values)

    end_index = count_consequtive_values[count_consequtive_values == length_of_dictionary].index[0]

    return end_index
    

def create_all_dictionaries_for_one_sheet(data : pd.DataFrame, level_pnl : str, level_bs : str, form : str) -> list:
    """
    Returns list dictionary of format {group name: list of codes} for a particular name, level, form from the raw data, 
    FOR ALL THE GROUPS OF DESIRED LEVEL IN ONE SHEET.
    
    Parameters:
    -----------
    data:pd.DataFrame
        Raw data from Деревья
    name:str
        Name of the group
    level_pnl:int
        Level of aggregation for PNL
    level_bs:int
        Level of aggregation for BS
    form:str
        Type of the form. "BS" for balance sheet, "PNL" for profit and loss
    """
    if form == "BS":
        level = level_bs
    elif form == "PNL":
        level = level_pnl
    else:
        print(f"There is no such form as {form} implemented yet, please choose either 'PNL' or 'BS'")
        return None
    group_names = create_level_names(preprocess_df(data, form), level)
    all_dicts = [None]*len(group_names)
    all_dicts = [create_dictionary_from_name_and_level(data, name, level, form) for name in group_names]
    return all_dicts



#способ добраться до положительного словаря выглядит так:
#list(all_dictionaries_dict["BS_old"][0].values())[0]["A"]
#здесь BS_old указывает на название листа группировки, первый ноль - номер группированного счета, второй ноль технический, 
#"A" - для положительного кода

def get_numbers(string : str) -> str:
    if len(re.findall("_", string)) > 0:
        return "REMOVED"
    if len(re.findall("\d+", string)) > 0:
        return re.findall("\d+", string)[0]
    else:
        return ""


def get_numbers_list(list_of_strings):
    get_numbers_vector = np.vectorize(get_numbers)
    list_of_numbers = get_numbers_vector(list_of_strings)
    list_of_numbers = list(list_of_numbers)
    return list_of_numbers
    
def positive_negative_dictionaries(big_dict : dict, form_name : str) -> tuple:
    
    """
    Returns a tuple, where first element is a dictionary of format {account : positive codes} and second element is {account : negative codes}. It also removes non-digital symbols in codes. !!! in future it seems desirable to separate this functionality to a separate object !!!
    
    Parameters:
    -----------
    big_dict::dict
        A dictionary created by create_all_dictionaries_for_one_sheet()
    form_name::str
        "BS_new", "BS_old", "PNL", "PNL_old", "PNL_very_old"
    """
    

    this_form_grouping_dictionary_positive = {}
    this_form_grouping_dictionary_negative = {}
    for number_of_account in range(len(big_dict[form_name])):
        if big_dict[form_name][number_of_account] is not None:
            name_of_account = list(big_dict[form_name][number_of_account].keys())[0]
            positive_value = list(big_dict[form_name][number_of_account].values())[0]["A"]
            negative_value = list(big_dict[form_name][number_of_account].values())[0]["P"]

            positive_value = list(map(str, positive_value))
            negative_value = list(map(str, negative_value))

            if len(positive_value) > 0:
                positive_value = get_numbers_list(positive_value)
            if len(negative_value) > 0:
                negative_value = get_numbers_list(negative_value)
            this_form_grouping_dictionary_positive[name_of_account] = positive_value
            this_form_grouping_dictionary_negative[name_of_account] = negative_value
    return this_form_grouping_dictionary_positive, this_form_grouping_dictionary_negative
    
    
def group_one_form(data : pd.DataFrame, form_name : str, big_dict : dict) -> pd.DataFrame:
    """
    Returns a table grouped in accordance with the grouping scheme provided in "form_name"
    
    Parameters:
    -----------
    data::pd.DataFrame
        Data for the banks. Must contain dates in index
    big_dict::dict
        A dictionary created by create_all_dictionaries_for_one_sheet()
    form_name::str
        "BS_new", "BS_old", "PNL", "PNL_old", "PNL_very_old"
    """
    
    grouping_dictionary_positive, grouping_dictionary_negative = \
                positive_negative_dictionaries(big_dict, form_name)
    
    if form_name in ["BS_new", "BS_old"]:
        data.NUM_SC = data.NUM_SC.apply(str)
    else:
        data.CODE = data.CODE.apply(str)
        
    if form_name in ["BS_new", "BS_old"]:
        positive_grouping = group(data=data, 
                                  aggschema=grouping_dictionary_positive, 
                                  form=101)
        negative_grouping = group(data=data, 
                                  aggschema=grouping_dictionary_negative, 
                                  form=101)
    else:
        positive_grouping = group(data=data, 
                                  aggschema=grouping_dictionary_positive, 
                                  form=102)
        negative_grouping = group(data=data, 
                                  aggschema=grouping_dictionary_negative, 
                                  form=102)    
        
    #Теперь объединим позитивные и негативные таблицы, заполним пропуски 
    # (там, где коды оказались только одного знака) нулями и просуммируем
    grouped_table = positive_grouping.merge(negative_grouping, 
                                            on = ["REGN", "new_code", "DT"], 
                                            how = "outer",
                                            suffixes = ("_positive", "_negative"))
    
    grouped_table = grouped_table.fillna(0)
    
    if form_name in ["BS_new", "BS_old"]:
        grouped_table["IITG"] = grouped_table.IITG_positive - grouped_table.IITG_negative
    else:
        grouped_table["IITG"] = grouped_table.SIM_ITOGO_positive - grouped_table.SIM_ITOGO_negative
        
    return grouped_table


class create_name_masks_container():
    
    def __init__(self, names_levels_pnl, names_levels_bs, pnl, bs, pnl_level, bs_level, additional_variables = []):
    #currently not all the names in the dictionary are real columns
    #luckily, not much of them
    #if I will ever fix this bug, I will deprecate the code below
        if (hasattr(pnl_level, '__len__') == False):
            self.pnl_mask = [i for i in np.array(names_levels_pnl[str(pnl_level)]) if i in np.array(df_pnl.columns)]
        else: 
            self.pnl_mask = []
            for level in pnl_level:
                self.pnl_mask = self.pnl_mask + [i for i in np.array(names_levels_pnl[str(level)]) if i in np.array(df_pnl.columns)]

        if (hasattr(bs_level, '__len__') == False):
            self.bs_mask = [i for i in np.array(names_levels_bs[str(bs_level)]) if i in np.array(df_bs.columns)]
        else: 
            self.bs_mask = []
            for level in bs_level:
                self.bs_mask = self.bs_mask + [i for i in np.array(names_levels_bs[str(level)]) if i in np.array(df_bs.columns)]

        self.pnl_encoding = [f"PNL{str(i)}" for i in range(1, len(self.pnl_mask) + 1)]
        self.bs_encoding = [f"BS{str(i)}" for i in range(1, len(self.bs_mask) + 1)]
        self.additional_variables = additional_variables
    
    def full_true_mask(self):
        return ["DT", "REGN"] + self.additional_variables + self.pnl_mask + self.bs_mask
    
    def pnl_index_mask(self):
        return ["DT", "REGN"] + self.additional_variables + self.pnl_mask
    
    def bs_index_mask(self):
        return ["DT", "REGN"] + self.additional_variables + self.bs_mask
    
    def encoding_mask(self):
        return ["DT", "REGN"] + self.additional_variables + self.pnl_encoding + self.bs_encoding
    
class create_days_container():
    
    def __init__(self, target_days):
        
        if (hasattr(target_days, '__len__') == False):
            self.target_days = [target_days]
        else:
            self.target_days = target_days
        
        self.target_names = [""]*len(self.target_days)
        for days_index in range(len(self.target_days)):
            self.target_names[days_index] = f"DefaultIn{self.target_days[days_index]}Days"
        
    def compare_days(self, number_of_days_real, number_of_days_benchmark):
        if pd.isnull(number_of_days_real):
            return 0
        else:
            return (int(number_of_days_real.days) <= int(number_of_days_benchmark))*1
        
    def create_target_columns(self, df, days_to_default_column = "DaysToDefault"):
        days_to_default_column = df[days_to_default_column]

        for days, days_name in zip(self.target_days, self.target_names):
            df[days_name] = days_to_default_column.apply(self.compare_days, number_of_days_benchmark = days)
        return df
    
    
def prepare_df_to_modelling(pnl, bs, defaults, name_masks, target_days, fillnan = 0):

    #create bs and pnl of needed level
    restricted_pnl = pnl[name_masks_container.pnl_index_mask()]
    restricted_bs = bs[name_masks_container.bs_index_mask()]
    
    #count NaN in each line
    restricted_bs["BSNan"] = restricted_bs.isnull().sum(axis = 1)
    restricted_pnl["PNLNan"] = restricted_pnl.isnull().sum(axis = 1)
    if fillnan == fillnan:
        restricted_pnl.fillna(fillnan, inplace = True)
        restricted_bs.fillna(fillnan, inplace = True)
    
    #merge bs and pnl
    merged_reporting = restricted_bs.merge(restricted_pnl, on = ["DT", "REGN"], how = "outer")
    merged_reporting = merged_reporting[name_masks_container.full_true_mask() + ["PNLNan", "BSNan"]]
    merged_reporting.columns = name_masks_container.encoding_mask() + ["PNLNan", "BSNan"] 

    #merge with defaults
    def func2(x):
        if type(x) == type("str"):
            return datetime.datetime.strptime(str(x), "%d.%m.%Y") 
        else:
            return x
        
    merged_reporting.REGN = merged_reporting.REGN.apply(str)
    
    pd.options.mode.chained_assignment = None
    defaults = defaults[defaults.DefaultType != "ликв."]
    defaults.DefaultDate = defaults.DefaultDate.apply(func2)
    merged_reporting = merged_reporting.merge(defaults, how = "left", on = "REGN")

    merged_reporting["DaysToDefault"] = merged_reporting.DefaultDate - merged_reporting.DT
    days_container_instance = create_days_container(target_days)

    merged_reporting = days_container_instance.create_target_columns(merged_reporting)
    
    not_needed_columns = np.array(defaults.columns)
    not_needed_columns = not_needed_columns[not_needed_columns != "REGN"]
    merged_reporting.drop(list(not_needed_columns) + ["DaysToDefault"], axis = 1, inplace = True)
    
    merged_reporting["Year"] = merged_reporting.DT.apply(lambda x: x.year)
    merged_reporting["Month"] = merged_reporting.DT.apply(lambda x: x.month)
    
    merged_reporting = merged_reporting.sort_values(by = ["REGN", "DT"])
    merged_reporting.fillna(method = "ffill", inplace = True)
    return merged_reporting

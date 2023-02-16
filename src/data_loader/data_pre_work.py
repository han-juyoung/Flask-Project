import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


DISCRETE_COLUMNS_BASIC = ['V1_USUG', 'V1_UPRO', 'V1_FHISTORY_DM', 'V1_preg_dm', 'goccp', 'gedu', 'bef_folt', 'v1_medu',
                        'bef_ink', 'bef_cont', 'preg_htn2', 'met_ope', 'V1_FHISTORY_HTN', 'pre_birth', 'aft_cal']
DIR_PATH = "./data/"
#Func 1: load 
#Import 2 files, file must exist in the same directory named 'data'
#- arguments -
#train = name of train set
#type_train = type of train set (ex: xlsx, csv)
#test = name of test set
#type_test = type of test set (ex: xlsx, csv)
#- returns -
#Return to 2 DataFrame formats(need pandas)
def load(train: str, type_train: str, test: str, type_test: str):
    data_train_path = DIR_PATH+ train + "." + type_train
    data_test_path = DIR_PATH + test + "." + type_test
    file_type_excel = ['xls', 'xlsx', 'xlsm', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']
    file_type_csv = 'csv'
    error_msg = 'Unavailable file format: '

    if type_train in file_type_excel:
        data_train = pd.read_excel(data_train_path)
    elif type_train is file_type_csv:
        data_train = pd.read_csv(data_train_path)
    else:
        print(error_msg + train + " -> " + type_train)
        exit(1)

    if type_test in file_type_excel:
        data_test = pd.read_excel(data_test_path)
    elif type_test is file_type_csv:
        data_test = pd.read_csv(data_test_path)
    else:
        print(error_msg + test + "-> " + type_test)
        exit(1)
    return data_train, data_test
    

#Func 2: synchronization
#Synchronize two DataFrames, by deleting non overlapping columns
#In this process, the y-values are seperated and returned
#--arguments-
#df1 = all columns will be deleted if the columns do not overlap with df2
#df2 = all columns will be deleted, if the columns do not overlap with df1
#--returns-
#df1_y = df1's y_datas
#df2_y = df2's y_datas
#There's no returns about dataframe, because already inplaced.

def synchronization (df1:pd.DataFrame, df2:pd.DataFrame):
    df1_y, df2_y = df1[['TARGET_GDM']], df2[['TARGET_GDM']]
    df2.rename(columns= {'AGE_NEW':'age'}, inplace=True)
    df1.drop(columns = ['no', 'class', 'site', 'TARGET_GDM'], inplace = True)
    df2.drop(columns = ['no', 'TARGET_GDM'], inplace = True)

    df2_columns_lower = list(df2.columns)
    df2_columns_lower = [i.lower() for i in df2_columns_lower]

    df1_columns = list(df1.columns)
    df1_columns = [i for i in df1_columns if i.lower() not in df2_columns_lower]
    df1.drop(columns = df1_columns, inplace=True)
    return df1_y, df2_y
    

#Func 3: drop_nan
#Replace the error value in the DataFrame with the nan value of numpy
#numpy must be installed
#-argument-
#df = DataFrame that you want to remove the error value
#-return-
#no return, just inplaced
def drop_nan(df: pd.DataFrame):
    columns = df.columns
    for i in columns:
        df[i].replace('.', np.nan, inplace = True)
        df[i].replace('\xa0 ', np.nan, inplace = True)
        df[i].replace('#NAME?', np.nan, inplace = True)


#Func 4: sorted_columns
#Sort columns in DataFrame in ascending order.
#-argument-
#df = DataFrame that you want to sort
#-return-
#sorted DataFrame
def sorted_columns(df: pd.DataFrame):
    return df.reindex(sorted(df.columns), axis=1)


#Func 5: imputation
#Fill in the nan vaulue of the DataFrame with the neighbors's values
#Sklearn must be installed and use the KNNImputer
#-arguments-
#df_train = DataFrame for training the Imputer (Pandas's DataFrame)
#df2 = Apply the Imputer learned by df_train to df2
#-returns-
#df_train = df_train that nan values have all been changed to appropriate values through the Imputer
#df2 = df2 that nan values have all been changed to appropriate values through the Imputer
def imputation(df_train: pd.DataFrame, df2: pd.DataFrame):
    imputer = KNNImputer(n_neighbors=5)
    columns = df_train.columns
    df_train = pd.DataFrame(imputer.fit_transform(df_train), columns=columns)
    df2 = pd.DataFrame(imputer.transform(df2), columns=columns)
    return df_train, df2


#Func 6: dicrete
#For elements that must have discrete values, it is annoying to have the values of Float.
#This Function changes all of those values into an integer form,
#and rounds out all the decimal points omitted in during process.
#-arguments-
#df = DataFrame that you want to organize discrete values
#columns = Columns that must have discrete elements, if you do not input, this module will use the pre-designated columns of KNU
#return
#df = DataFrame that organized discrete values 
def discrete( df: pd.DataFrame, columns=DISCRETE_COLUMNS_BASIC):
    df[columns] = df[columns].round(0).astype('int')
    return df


#Func 7: standard_scale
#Scailing the DataFrame using StandardScaler
#Sklearn must be installed and use the KNNImputer
#-argumets-
#df = DataFrame for training the StandardScaler
#df2 = Apply the StandardScaler learned by df to df2
#-returns-
#df = DataFrame 'df' that scaled using StandardScaler
#df2 = DataFrame 'df2' that scaled using StandardScaler
def standard_scale(df: pd.DataFrame, df2:pd.DataFrame):
    standard_scaler = StandardScaler()
    columns = df.columns
    df = pd.DataFrame(standard_scaler.fit_transform(df), columns=columns)
    df2 = pd.DataFrame(standard_scaler.transform(df2), columns=columns)
    return df, df2

#Func 8: merge_xy
#Merge the DataFrame for X values and y values into one DataFrame and return
#-arguments-
#df_x = DataFrame that has X values
#df_y =  DataFrame that has y values
#-return-
#Merged DataFrame
def merge_xy(df_x:pd.DataFrame, df_y:pd.DataFrame):
    return df_y.join(df_x)

#Func 9: integrated_module
#From Func1 to Func8, preprocess the DataFrame and save it as an Excel file
#If an error occurs, i recommend that run it in order from Func1
#Two DataFrame are required, the first one is used to train the imputer and scaler
#--argumets--
#same with Func1
def integrated_module(train: str, type_train: str, test: str, type_test: str):
    train_data_file_name = DIR_PATH + train +"_pre_worked." + type_train  
    test_data_file_name = DIR_PATH + test +  "_pre_worked." +type_test  
    df, df2 = load(train, type_train, test, type_test)
    df_y, df2_y = synchronization(df, df2)
    drop_nan(df)
    drop_nan(df2)
    df, df2 = sorted_columns(df), sorted_columns(df2) 
    df, df2 = imputation(df, df2)
    df, df2 = discrete(df), discrete(df2)
    df, df2 = standard_scale(df, df2)
    df, df2 = merge_xy(df, df_y), merge_xy(df2, df2_y)
    df.to_excel(train_data_file_name, index=False)
    df2.to_excel(test_data_file_name, index=False) 

    print("Success, Check the directory")

if __name__ == '__main__':
    integrated_module('train_set', 'xlsx', 'test_set', 'xlsx')

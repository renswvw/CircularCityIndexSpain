# 01. Data Management (Pipeline)

# Import modules
from pathlib import Path
import os
import geopandas as gpd
import glob
from glob import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterstats
from rasterio.plot import show
import re
from fuzzywuzzy import process, fuzz
from sklearn.impute import KNNImputer, SimpleImputer
import xlrd
import openpyxl
import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def DataManagement():
    # papermill parameters cell
    OUTPUT_WARNINGS = False
    SAVE_INTERIMDATA = True
    FILL_NAN_W_0 = False
    FILL_NAN_W_KNN = True

    if OUTPUT_WARNINGS is False:
        import warnings

        warnings.filterwarnings("ignore")

    ### Data
    # Data folders
    RAW_FOLDER = "data/raw/"
    INTERIM_FOLDER = "data/interim/"
    ADMBOUND_RAW_FOLDER = "data/raw/AdmBound_rawdata/"
    ADMBOUND_INTERIM_FOLDER = 'data/interim/AdmBound_interimdata/' 
    ECR4_TEMPORAL_FOLDER = "data/interim/ECR4_temporaldata/"
    ECR5_TEMPORAL_FOLDER = "data/interim/ECR5_temporaldata/"
    W2_RAW_FOLDER = "data/raw/W2_rawdata/"
    W3_RAW_FOLDER = "data/raw/W3_rawdata/"
    DEMOGRAPHIC_RAW_FOLDER = "data/raw/demographic_rawdata/"
    INTERIM_DEMOGRAPHIC_FOLDER = "data/interim/demographic_interimdata/"

    # create folders if not exists
    Path(RAW_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INTERIM_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(ADMBOUND_RAW_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(ADMBOUND_INTERIM_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(ECR4_TEMPORAL_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(ECR5_TEMPORAL_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(W2_RAW_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(W3_RAW_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DEMOGRAPHIC_RAW_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INTERIM_DEMOGRAPHIC_FOLDER).mkdir(parents=True, exist_ok=True)

    # Datasets KPI
    ADMBOUND_RAWDATA_GML = ADMBOUND_RAW_FOLDER + 'au_AdministrativeUnit_4thOrder0.gml'
    ADMBOUND_INTERIMDATA = INTERIM_FOLDER + 'AdmBound_interimdata/AdmBound_interimdata.shp'
    POP21_RAWDATA = RAW_FOLDER + 'POP21_rawdata.xlsx'
    POP21_INTERIMDATA = INTERIM_FOLDER + 'POP21_interimdata.csv'
    D1_RAWDATA = RAW_FOLDER + 'D1_rawdata.xlsx'
    D2_RAWDATA = RAW_FOLDER + 'D2_rawdata.xlsx'
    D3_RAWDATA = RAW_FOLDER + 'D3_rawdata.xlsx'
    D4_RAWDATA = RAW_FOLDER + 'D4_rawdata.xlsx'
    ECR1_RAWDATA = RAW_FOLDER + 'ECR1_rawdata.xlsx'
    ECR1_INTERIMDATA = INTERIM_FOLDER + 'ECR1_interimdata.csv'
    ECR2_RAWDATA = RAW_FOLDER + 'ECR2_rawdata.xlsx'
    ECR4_RAWDATA = RAW_FOLDER + 'ECR4_rawdata.tif'
    ECR5_RAWDATA = RAW_FOLDER + 'ECR5_rawdata.tif'
    M1_RAWDATA = RAW_FOLDER + 'M1_rawdata.csv'
    M2_RAWDATA = RAW_FOLDER + 'M2_rawdata.csv'
    M3_RAWDATA = RAW_FOLDER + 'M3_rawdata.csv'
    M4_RAWDATA = RAW_FOLDER + 'M4_rawdata.csv'

    # Datasets Demographic features
    INCOME_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'INCOME_rawdata.xlsx'
    DEMOGRAPHIC_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'DEMOGRAPHIC_rawdata.xlsx'
    GINI_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'GINI_rawdata.xlsx'
    DEM_GROWTH_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'DEM_GROWTH_rawdata.xlsx'
    ECONOMIC_COMPANY_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'ECONOMIC_COMPANY_rawdata.xlsx'
    TOURISM_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'TOURISM_rawdata.xls'
    DEBT_MUNICIPALITY_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'DEBT_MUNICIPALITY_rawdata.xlsx'
    POP_GENDER_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'POP_GENDER_rawdata.xlsx'
    RESIDENTIAL_BUILDINGS_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'RESIDENTIAL_BUILDINGS_rawdata.xlsx'
    TOURIST_HOUSES_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'TOURIST_HOUSES_rawdata.xlsx'
    UNEMPLOYMENT_BENEFITS_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'UNEMPLOYMENT_BENEFITS_rawdata.xlsx'
    AGRICULTURE_RAWDATA = DEMOGRAPHIC_RAW_FOLDER + 'AGRICULTURE_rawdata.xlsx'


    # PART 1 - GENERAL MUNICIPAL DATA
    ## Administrative Boundaries
    # read as GeoDataFrame
    shp_municipality = gpd.read_file(ADMBOUND_RAWDATA_GML)

    # Define Autonomous Community Code (CAUC), Province Code (CPRO), and Municipality Code (CMUN)
    shp_municipality['CAUC'] = shp_municipality['nationalCode'].astype("string").apply(lambda x : x[2:4])
    shp_municipality['CPRO'] = shp_municipality['nationalCode'].astype("string").apply(lambda x : x[6:8])
    shp_municipality['CMUN'] = shp_municipality['nationalCode'].astype("string").apply(lambda x : x[8:])

    # Define total code for Municipality (CTOT)
    shp_municipality['CTOT'] = shp_municipality['CAUC'] + shp_municipality['CPRO'] + shp_municipality['CMUN']

    # filter columns
    shp_municipality = shp_municipality[[
    'CTOT',
    'text',
    'geometry']]

    # create a dictionary
    # key = old name
    # value = new name
    dict = {'text': 'Municipality'}
    
    # call rename () method
    shp_municipality.rename(columns=dict,
            inplace=True)

    # Data homogenalization - Title case to all column names
    shp_municipality['Municipality'] = shp_municipality['Municipality'].str.title()

    # Data validation - check if each municipality has its own code
    ids = shp_municipality['Municipality']
    shp_municipality[ids.isin(ids[ids.duplicated()])].sort_values("Municipality")

    # exports the dataframe into Shapefile and GeoPackage file with
    file_name = 'AdmBound_interimdata' 
    data_format_shp = '.shp'
    data_format_gpkg = '.gpkg'
    export_name = file_name

    if SAVE_INTERIMDATA is True:
        shp_municipality.to_file(ADMBOUND_INTERIM_FOLDER + export_name + data_format_shp)
        shp_municipality.to_file(ADMBOUND_INTERIM_FOLDER + export_name + data_format_gpkg)

    ## Population Register
    # read the raw excel file and merge all sheets into dataframe
    excelFile = pd.ExcelFile(POP21_RAWDATA)
    df = pd.concat(pd.read_excel(excelFile, sheet_name=None), ignore_index=True)

    # Replace column headers with second row and drop first two rows
    df.columns = df.iloc[1]
    df = df[2:]

    # Create column for Municipality Code Total (CTOT)
    df['CTOT'] = df['CPRO'].astype('str') + df['CMUN'].astype('str')

    # filter columns that refer to municipality name and population
    df = df[[
    'CTOT',
    'POB21']]

    # create a dictionary
    # key = old name
    # value = new name
    dict = {'CTOT':'CMUN',
        'POB21': 'POP21'}
    
    # call rename () method
    df.rename(columns=dict,
            inplace=True)

    # exports the dataframe into csv file with
    file_name = 'POP21_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df.to_csv(INTERIM_FOLDER + export_name, index=False)

    # PART 2 - KPI DATA
    ## D1
    # read the raw excel file from the url
    excelFile = pd.ExcelFile(D1_RAWDATA)
    df = pd.read_excel(excelFile, 'EELL')

    # Clean datasets
    # Remove first 5 rows
    df.drop(df.index[:5], inplace=True)

    # Drop rows where municipality is nan
    df = df[df['Unnamed: 3'].notna()]

    # Data Homogenization 
    # Remove first 3 characters and last 1 characters from CMUN
    df['CMUN'] = df['Unnamed: 3'].str.replace('L', '').str[2:].str[:-1]

    # Add 1 to all municipalities that are in list
    df['D1'] = 1

    # Select columns
    df = df[[
    'CMUN',
    'D1']]

    # drop duplicate rows
    df = df.drop_duplicates(subset=['CMUN'], keep='first')

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Get all CMUN codes of municipalities
    shp_municipality['CMUN'] = shp_municipality['CTOT'].str[2:]

    # Merge D1 dataset with all CMUN municipality dataset
    df_D1_merge = df.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0
    df_D1_merge['D1'] = df_D1_merge['D1'].fillna(0)

    df_D1 = df_D1_merge[['CTOT', 'D1']]

    # exports the dataframe into csv file with
    file_name = 'D1_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_INTERIMDATA is True:
        df_D1.to_csv(INTERIM_FOLDER + export_name, index=False)

    ## D2
    # read the raw excel file from the url
    excelFile = pd.ExcelFile(D2_RAWDATA)
    df = pd.read_excel(excelFile, 'EELL')

    # Clean datasets
    # Remove first 5 rows
    df.drop(df.index[:5], inplace=True)

    # Drop rows where municipality is nan
    df = df[df['Unnamed: 3'].notna()]

    # Data Homogenization 
    # Remove first 3 characters and last 1 characters from CMUN
    df['CMUN'] = df['Unnamed: 3'].str.replace('L', '').str[2:].str[:-1]

    # Select all rows with value 'CLAVE' (Cl@ve)
    df = df[df['Servicios de la SGAD. Organismos Usuarios   (actualización diaria). Administración Local'] == 'CLAVE']

    # Give value 1 to all rows that are selected
    df['D2'] = 1

    # Select columns
    df = df[[
    'CMUN',
    'D2']]

    # drop duplicate rows
    df = df.drop_duplicates(subset=['CMUN'], keep='first')

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Get all CTOT codes of municipalities
    shp_municipality['CMUN'] = shp_municipality['CTOT'].str[2:]

    # Merge D2 with CMUN municipality dataset
    df_D2_merge = df.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0
    df_D2_merge['D2'] = df_D2_merge['D2'].fillna(0)

    df_D2 = df_D2_merge[['CTOT', 'D2']]

    # exports the dataframe into csv file with
    file_name = 'D2_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_INTERIMDATA is True:
        df_D2.to_csv(INTERIM_FOLDER + export_name, index=False)

    ## D3
    # read the raw excel file from the url
    excelFile = pd.ExcelFile(D3_RAWDATA)
    df = pd.read_excel(excelFile, 'Municipio')

    # Add extra digit to CMUN if it contains less than 5 characters
    df['CMUN'] = df['CMUN'].apply(lambda x: '{0:0>5}'.format(x))

    # filter columns that refer to 'cobertura 30Mpbs' (see leyenda)
    df = df[[
    'CMUN',
    'Cob. 30Mbps\n(junio 2021)']]

    # create a dictionary
    dict = {'Cob. 30Mbps\n(junio 2021)': 'D3'}
    
    # call rename () method
    df.rename(columns=dict,
            inplace=True)

    # exports the dataframe into csv file with
    file_name = 'D3_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_INTERIMDATA is True:
        df.to_csv(INTERIM_FOLDER + export_name, index=False)

    ## D4
    # read the raw excel file from the url
    excelFile = pd.ExcelFile(D4_RAWDATA)
    df = pd.read_excel(excelFile, 'EELL')

    # Clean datasets
    # Remove first 5 rows
    df.drop(df.index[:5], inplace=True)

    # Drop rows where municipality is nan
    df = df[df['Unnamed: 3'].notna()]

    # Select all rows with value 'Ayuntamiento'
    df = df[df['Unnamed: 4'].str.contains('Ayuntamiento')==True]

    # Data Homogenization 
    # Remove first 3 characters and last 1 characters from CMUN
    df['CMUN'] = df['Unnamed: 3'].str.replace('L', '').str[2:].str[:-1]

    # Copy column
    df['Services'] = df['Servicios de la SGAD. Organismos Usuarios   (actualización diaria). Administración Local']

    # Select all rows without value 'SIA' or 'CLAVE' (Cl@ve)
    df = df[df["Services"].str.contains('SIA|CLAVE')==False]

    # Give value (1/total of unique service) to all rows that are selected
    df['D4_service'] = 1/(df['Services'].nunique())

    # Drop rows with double values for both CMUN and type of Services
    df = df.drop_duplicates(subset=['CMUN', 'Services'], keep='last')

    # Sum values of D4 per CMUN
    df['D4'] = df.groupby(['CMUN'])['D4_service'].transform('sum')

    # Select columns
    df = df[[
    'CMUN',
    'D4']]

    # drop duplicate rows
    df = df.drop_duplicates(subset=['CMUN'], keep='first')

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Get all CTOT codes of municipalities
    shp_municipality['CMUN'] = shp_municipality['CTOT'].str[2:]

    # Merge D4 with CMUN municipality dataset
    df_D4_merge = df.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0
    df_D4_merge['D4'] = df_D4_merge['D4'].fillna(0)

    df_D4 = df_D4_merge[['CTOT', 'D4']]

    # exports the dataframe into csv file with
    file_name = 'D4_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_INTERIMDATA is True:
        df_D4.to_csv(INTERIM_FOLDER + export_name, index=False)

    ## ECR1 & ECR2

    # read the raw excel file from the raw folder
    excel_file = ECR1_RAWDATA
    df = pd.read_excel(excel_file)

    # Data cleaning
    # filter columns that refer to 'Signatories'
    df = df[['Signatories']]

    # create a dictionary
    dict = {'Signatories': 'Municipality'}
    
    # call rename () method
    df.rename(columns=dict,
            inplace=True)

    # Data Homogenization 
    # remove 'ES' from values and title-format to values
    df['Municipality'] = df['Municipality'].str.replace(r', ES', '').str.title()

    # read the ECR2_RAWDATA excel file per sheet
    df_adaption = pd.read_excel(ECR2_RAWDATA, 'Commitment - Adaption')
    df_2020 = pd.read_excel(ECR2_RAWDATA, 'Commitment - 2020')
    df_2030 = pd.read_excel(ECR2_RAWDATA, 'Commitment - 2030')
    df_2050 = pd.read_excel(ECR2_RAWDATA, 'Commitment - 2050')

    # filter to municipality name and change values to title-format
    df_adaption = df_adaption['Signatories'].str.title()
    df_2020 = df_2020['Signatories'].str.title()
    df_2030 = df_2030['Signatories'].str.title()
    df_2050 = df_2050['Signatories'].str.title()

    # Create new columns, based on each seperate dataframe per level of commitment
    df['Commitment_Adaption'] = df['Municipality'].apply(lambda str: any([(reqWord in str) for reqWord in df_adaption]))
    df['Commitment_2020'] = df['Municipality'].apply(lambda str: any([(reqWord in str) for reqWord in df_2020]))
    df['Commitment_2030'] = df['Municipality'].apply(lambda str: any([(reqWord in str) for reqWord in df_2030]))
    df['Commitment_2050'] = df['Municipality'].apply(lambda str: any([(reqWord in str) for reqWord in df_2050]))

    # Assign value to all municipalities in list by creating condition for dataset merging later on
        #['Commitment_Adaption'] = 0.1
        #['Commitment_2020'] = 0.3
        #['Commitment_2030'] = 0.6
        #['Commitment_2050'] = 0.9

    conditions_Adaption = [
        (df['Commitment_Adaption'] == True),
        (df['Commitment_Adaption'] == False)] 

    conditions_2020 = [
        (df['Commitment_2020'] == True),
        (df['Commitment_2020'] == False)]    

    conditions_2030 = [
        (df['Commitment_2030'] == True),
        (df['Commitment_2030'] == False)]  

    conditions_2050 = [
        (df['Commitment_2050'] == True),
        (df['Commitment_2050'] == False)]  

    # Create a list of values that must be assigned when condition is true or false
    values_Adaption = [1, 0]
    values_2020 = [2, 0]
    values_2030 = [4, 0]
    values_2050 = [10, 0]

    # Create new column with binary assigning
    df['Commitment_Adaption_Score'] = np.select(conditions_Adaption, values_Adaption)
    df['Commitment_2020_Score'] = np.select(conditions_2020, values_2020)
    df['Commitment_2030_Score'] = np.select(conditions_2030, values_2030)
    df['Commitment_2050_Score'] = np.select(conditions_2050, values_2050)

    # Sum of all scores to new total score in Column
    df['ECR2'] =  df['Commitment_Adaption_Score'] + df['Commitment_2020_Score'] + df['Commitment_2030_Score'] + df['Commitment_2050_Score']

    # Maximise output score to 1
    df.loc[df['ECR2'] == 1, 'ECR2'] = 0.1
    df.loc[df['ECR2'] == 2, 'ECR2'] = 0.3
    df.loc[df['ECR2'] == 3, 'ECR2'] = 0.4
    df.loc[df['ECR2'] == 4, 'ECR2'] = 0.6
    df.loc[df['ECR2'] == 5, 'ECR2'] = 0.7
    df.loc[df['ECR2'] == 6, 'ECR2'] = 0.6
    df.loc[df['ECR2'] == 7, 'ECR2'] = 0.7
    df.loc[df['ECR2'] == 10, 'ECR2'] = 0.9
    df.loc[df['ECR2'] == 11, 'ECR2'] = 1
    df.loc[df['ECR2'] == 12, 'ECR2'] = 0.9
    df.loc[df['ECR2'] == 13, 'ECR2'] = 1
    df.loc[df['ECR2'] == 14, 'ECR2'] = 0.9
    df.loc[df['ECR2'] == 15, 'ECR2'] = 1
    df.loc[df['ECR2'] == 16, 'ECR2'] = 0.9
    df.loc[df['ECR2'] == 17, 'ECR2'] = 1

    # filter columns that refer to 'ECR2' 
    df = df[[
    'Municipality',
    'ECR2']]

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)
    shp_names = shp_municipality['Municipali']

    # DATA CLEANING
    # Create new column to append fuzzywuzzy to
    fuzzywuzzy_ECR2 = pd.DataFrame(columns=['ECR2_name','ECR2','Match_name', 'CTOT', 'score', 'fuzzywuzzy_method'])

    # Apply fuzzywuzzy to link municipality names for merging
    n=0

    # try fuzz.token_sort_ratio
    for df_name in df['Municipality']:
        print("Search name : ", df_name, '\n', 62*'-', '\n')
        print(n, "/", len(df.axes[0]))
        
        choices = shp_names
        result = process.extract(df_name, choices, scorer=fuzz.token_sort_ratio)        

        ECR2 = df['ECR2'].iloc[n]
        match = result[0][0]
        score = result[0][1]
        index = result[0][2]
        CTOT = shp_municipality['CTOT'].iloc[index]

        if score > 90:
            new_row = {'ECR2_name':df_name, 
                        'ECR2':ECR2,
                        'Match_name':match,                    
                        'CTOT':CTOT,
                        'score':score,
                        'fuzzywuzzy_method': 'token_sort_ratio'}

        # try fuzz.WRatio 
        else:
            result_else = process.extract(df_name, choices, scorer=fuzz.WRatio)

            match_else = result_else[0][0]
            score_else = result_else[0][1]
            index_else = result_else[0][2]
            CTOT_else = shp_municipality['CTOT'].iloc[index_else]

            if score_else > 90:
                new_row = {'ECR2_name':df_name, 
                    'ECR2':ECR2,
                    'Match_name':match_else,
                    'CTOT':CTOT_else,
                    'score':score_else,
                    'fuzzywuzzy_method': 'WRatio'}

            #try fuzz.token_set_ratio --> ignores duplicated words
            else:
                result_token_set_ratio = process.extract(df_name, choices, scorer=fuzz.token_set_ratio)

                match_token_set_ratio = result_token_set_ratio[0][0]
                score_token_set_ratio = result_token_set_ratio[0][1]
                index_token_set_ratio = result_token_set_ratio[0][2]
                CTOT_token_set_ratio = shp_municipality['CTOT'].iloc[index_token_set_ratio]

                if score_token_set_ratio > 90:
                    new_row = {'ECR2_name':df_name, 
                        'ECR2':ECR2,
                        'Match_name':match_token_set_ratio,
                        'score':score_token_set_ratio,
                        'CTOT':CTOT_token_set_ratio,
                        'fuzzywuzzy_method': 'token_set_ratio'}
                
                # If no score above 90, return NAN-value
                else:
                    new_row = {'ECR2_name':df_name, 
                        'ECR2':ECR2,
                        'Match_name':np.NAN,                    
                        'CTOT':np.NAN,
                        'score':np.NAN,
                        'fuzzywuzzy_method': np.NAN}


        fuzzywuzzy_ECR2 = fuzzywuzzy_ECR2.append(new_row, ignore_index=True)

        n = n+1

    df_ECR2 = fuzzywuzzy_ECR2[[
    'CTOT',
    'ECR2']]

    # drop duplicate rows
    df_ECR2 = df_ECR2.drop_duplicates(subset=['CTOT'], keep='first')

    # Create ECR1 out of ECR2
    df_ECR1 = df_ECR2.copy()

    # Assign value 'Yes' to all municipalities in list (1 = signed CoM)
    df_ECR1['ECR1'] = 1

    # filter columns that refer to 'Signed_CovenantofMayors_Binary'
    df_ECR1 = df_ECR1[[
    'CTOT',
    'ECR1']]

    # Get all CTOT codes of municipalities
    shp_CTOT = shp_municipality['CTOT']

    # Merge ECR1 and ECR2 dataset with all CTOT municipality dataset
    df_ECR1_merge = df_ECR1.merge(shp_municipality, how = 'right')
    df_ECR2_merge = df_ECR2.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0 (if municipality does not have punto limpio)
    df_ECR1_merge['ECR1'] = df_ECR1_merge['ECR1'].fillna(0)
    df_ECR2_merge['ECR2'] = df_ECR2_merge['ECR2'].fillna(0)

    df_ECR1 = df_ECR1_merge[['CTOT', 'ECR1']]
    df_ECR2 = df_ECR2_merge[['CTOT', 'ECR2']]

    # export the ECR1 dataframe into csv file
    file_name = 'ECR1_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_ECR1.to_csv(INTERIM_FOLDER + export_name, index=False)

    # export the ECR2 dataframe into csv file
    file_name = 'ECR2_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_ECR2.to_csv(Path(INTERIM_FOLDER) / export_name, index=False)

    ## ECR4

    # create path for temporal dataset
    ECR4_TEMPORALDATA = ECR4_TEMPORAL_FOLDER + "interim_PM10_avg19.tif"

    # replace nodata value raster
    with rasterio.open(ECR4_RAWDATA, "r+") as src:
        src.nodata = 0 # set the nodata value
        profile = src.profile
        profile.update(
                dtype=rasterio.uint8,
                compress='lzw'
        )

        with rasterio.open(ECR4_TEMPORALDATA, 'w',  **profile) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                # band = np.where(band!=1,0,band) # if value is not equal to 1 assign no data value i.e. 0
                band = np.where(band==0,0,band) # for completeness
                dst.write(band,i)

    # read shapefile Administrative Boundaries Spain
    boundaries = gpd.read_file(ADMBOUND_INTERIMDATA)
    print(boundaries.crs)

    # read rasterfile PM10
    rf = rasterio.open(ECR4_TEMPORALDATA)

    # check characteristics and CRS of interim raster file
    print('width =',(rf.width))
    print('height =',(rf.height))
    print('BoundingBox =',(rf.bounds))
    print('CRS =',(rf.crs))

    # change projection
    CRS = rf.crs
    boundaries = boundaries.to_crs(CRS)

    # plotting rasterfile and shapefile together - to check if overlay succeeded
    fig, ax = plt.subplots(1,1)
    show(rf, ax=ax, title = 'PM10 by administrative districs')
    boundaries.plot(ax=ax, facecolor='None', edgecolor = 'yellow')
    plt.show()

    # Extract raster values to a numpy nd array
    PM10_array = rf.read(1)
    type(PM10_array)

    # Transform raster
    affine = rf.transform

    # Get metadata
    metadata = rf.meta
    print('Metadata =',metadata)
    print('NoData =',rf.nodata)
    print('CRS =',rf.crs)

    # Calculate zonal stats
    average_PM10 = rasterstats.zonal_stats(boundaries, PM10_array, affine = affine, stats = ['mean'], geojson_out = True)

    # Extracting PM10 values per municipality
    mean_PM10 = []
    i = 0

    while i < len(average_PM10):
        mean_PM10.append(average_PM10[i]['properties'])
        i = i + 1

    # Extracting data to dataframe
    df = pd.DataFrame(mean_PM10)

    # Rename column headers
    df['ECR4_temp'] = df['mean']

    # Select only the columns with municipality name and value
    df_ECR4 = df[['CTOT','ECR4_temp']]

    # ADD KNN IMPUTER
    # Replace NaN-values with k-nearest neighbour value
    if FILL_NAN_W_KNN is True:
        df_index = df_ECR4.set_index('CTOT')
        df_transformed = df_index.copy()
        df_transformed = df_transformed.values.reshape(-1, 1)
        if pd.isnull(df_transformed).sum().sum() != 0:
            knn_ECR4 = KNNImputer()
            knn_ECR4.fit(df_transformed)
            df_transformed = knn_ECR4.transform(df_transformed)
        else:
            pass

        # Extra check: if df_transformed still contains NaN-values, then replace with 'mean' value
        if pd.isnull(df_transformed).sum().sum() != 0:
            simple_y = SimpleImputer(missing_values=np.nan, strategy='mean')
            simple_y.fit(df_transformed)
            df_transformed = simple_y.transform(df_transformed)
        else: 
            pass

        # Create function to transform dataframe with index, column headers
        df_ECR4['ECR4'] = pd.DataFrame(df_transformed, columns = ['ECR4'])
        df_ECR4 = df_ECR4.drop(['ECR4_temp'], axis=1)

    # exports the dataframe into csv file with
    file_name = 'ECR4_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_ECR4.to_csv(INTERIM_FOLDER + export_name, index=False)

    # delete temporal folder (ECR4_TEMPORAL_FOLDER)
    shutil.rmtree(ECR4_TEMPORAL_FOLDER, ignore_errors=True)

    ## ECR5

    # create path for temporal dataset
    ECR5_TEMPORALDATA = ECR5_TEMPORAL_FOLDER + "nox_avg19.tif"

    # replace nodata value raster
    with rasterio.open(ECR5_RAWDATA, "r+") as src:
        src.nodata = 0 # set the nodata value
        profile = src.profile
        profile.update(
                dtype=rasterio.uint8,
                compress='lzw'
        )

        with rasterio.open(ECR5_TEMPORALDATA, 'w',  **profile) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                # band = np.where(band!=1,0,band) # if value is not equal to 1 assign no data value i.e. 0
                band = np.where(band==0,0,band) # for completeness
                dst.write(band,i)

    # read shapefile Administrative Boundaries Spain
    boundaries = gpd.read_file(ADMBOUND_INTERIMDATA)
    print('ADMBOUND_INTERIMDATA --> CRS =',(boundaries.crs))

    # read rasterfile NOx
    rf = rasterio.open(ECR5_TEMPORALDATA)

    # check interim raster file
    print('width =',(rf.width))
    print('height =',(rf.height))
    print('BoundingBox =',(rf.bounds))
    print('CRS =',(rf.crs))

    # change projection
    CRS = rf.crs
    boundaries = boundaries.to_crs(CRS)

    # plotting rasterfile and shapefile together - to check if overlay succeeded
    fig, ax = plt.subplots(1,1)
    show(rf, ax=ax, title = 'NOx by administrative districs')
    boundaries.plot(ax=ax, facecolor='None', edgecolor = 'yellow')
    plt.show()

    # Extract raster values to a numpy nd array
    NOx_array = rf.read(1)
    type(NOx_array)

    # Transform raster
    affine = rf.transform

    # Get metadata
    metadata = rf.meta
    print('Metadata =',metadata)
    print('NoData =',rf.nodata)
    print('CRS =',rf.crs)

    # Calculate zonal stats
    average_NOx = rasterstats.zonal_stats(boundaries, NOx_array, affine = affine, stats = ['mean'], geojson_out = True)

    # Extracting NOx values per municipality
    mean_NOx = []
    i = 0

    while i < len(average_NOx):
        mean_NOx.append(average_NOx[i]['properties'])
        i = i + 1

    # Extracting data to dataframe
    df = pd.DataFrame(mean_NOx)

    # Rename column headers
    df['ECR5_temp'] = df['mean']

    # Select only the columns with municipality name and value
    df_ECR5 = df[['CTOT','ECR5_temp']]

    # ADD KNN IMPUTER
    # Replace NaN-values with k-nearest neighbour value
    if FILL_NAN_W_KNN is True:
        df_index = df_ECR5.set_index('CTOT')
        df_transformed = df_index.copy()
        df_transformed = df_transformed.values.reshape(-1, 1)
        if pd.isnull(df_transformed).sum().sum() != 0:
            knn_ECR5 = KNNImputer()
            knn_ECR5.fit(df_transformed)
            df_transformed = knn_ECR5.transform(df_transformed)
        else:
            pass

        # Extra check: if df_transformed still contains NaN-values, then replace with 'mean' value
        if pd.isnull(df_transformed).sum().sum() != 0:
            simple_y = SimpleImputer(missing_values=np.nan, strategy='mean')
            simple_y.fit(df_transformed)
            df_transformed = simple_y.transform(df_transformed)
        else: 
            pass

        # Create function to transform dataframe with index, column headers
        df_ECR5['ECR5'] = pd.DataFrame(df_transformed, columns = ['ECR5'])
        df_ECR5 = df_ECR5.drop(['ECR5_temp'], axis=1)

    # exports the dataframe into csv file with
    file_name = 'ECR5_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_ECR5.to_csv(INTERIM_FOLDER + export_name, index=False)

    # delete temporal folder (ECR5_TEMPORAL_FOLDER)
    #shutil.rmtree(ECR5_TEMPORAL_FOLDER, ignore_errors=False, onerror=None)

    ## M1

    # read datasets into dataframe
    df_M1_raw = pd.read_csv(M1_RAWDATA)
    df_POP21 = pd.read_csv(POP21_INTERIMDATA)

    # DATA HOMOGENIZATION
    # Add extra digit to CCI POP21 CMUN - if it contains less than 5 characters
    df_POP21['CMUN'] = df_POP21['CMUN'].apply(lambda x: '{0:0>5}'.format(x))

    # Add extra digit to CCI M1 CTOT - if it contains less than 7 characters
    df_M1_raw['CTOT'] = df_M1_raw['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Creat for CCI M1 column CMUN out of CTOT
    df_M1_raw['CMUN'] = df_M1_raw['CTOT'].str[2:]

    # Merge datasets of M1 and POP21 into one dataframe
    df_M1 = df_M1_raw.merge(df_POP21, how = 'right')

    # create new column for relative number of pedestrian areas per 100 inhabitants
    df_M1['M1'] = (df_M1['M1_surface'] / df_M1['POP21']) * 100

    # Replace NaN values with 0 in dataframe
    if FILL_NAN_W_0 is True:
        df_M1['M1'] = df_M1['M1'].fillna(0)

    if FILL_NAN_W_0 is False:
        df_M1['M1'] = df_M1['M1'].replace(0.0, np.nan, inplace=False)

    # filter columns that refer to KPIs
    df_M1 = df_M1[[
        'CMUN', 
        'M1']]

    # exports the dataframe into csv file
    file_name = 'M1_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_M1.to_csv((INTERIM_FOLDER) + export_name, index=False)

    ## M2

    # read datasets into dataframe
    df_M2_raw = pd.read_csv(M2_RAWDATA)
    df_POP21 = pd.read_csv(POP21_INTERIMDATA)

    # DATA HOMOGENIZATION
    # Add extra digit to CCI POP21 CMUN - if it contains less than 5 characters
    df_POP21['CMUN'] = df_POP21['CMUN'].apply(lambda x: '{0:0>5}'.format(x))

    # Add extra digit to CCI M2 CTOT - if it contains less than 7 characters
    df_M2_raw['CTOT'] = df_M2_raw['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Creat for CCI M2 column CMUN out of CTOT
    df_M2_raw['CMUN'] = df_M2_raw['CTOT'].str[2:]

    # Merge datasets of M2 and POP21 into one dataframe
    df_M2 = df_M2_raw.merge(df_POP21, how = 'right')

    # create new column for relative number of charging_stations per 1,000 inhabitants
    df_M2['M2'] = (df_M2['M2_absolute'] / df_M2['POP21']) * 1000

    # Replace NaN values with 0 in dataframe
    if FILL_NAN_W_0 is True:
        df_M2['M2'] = df_M2['M2'].fillna(0)

    # Drop duplicate rows
    df_M2 = df_M2.drop_duplicates(subset=['CMUN'])

    # filter columns that refer to KPIs
    df_M2 = df_M2[[
        'CMUN', 
        'M2']]

    # exports the dataframe into csv file
    file_name = 'M2_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_M2.to_csv((INTERIM_FOLDER) + export_name, index=False)

    ## M3

    # read datasets into dataframe
    df_M3_raw = pd.read_csv(M3_RAWDATA)
    gdf_AdmBound = gpd.read_file(ADMBOUND_INTERIMDATA)

    # DATA HOMOGENIZATION
    # Add extra digit to CCI M3 CTOT - if it contains less than 7 characters
    df_M3_raw['CTOT'] = df_M3_raw['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Creat for CCI M3 column CMUN out of CTOT
    df_M3_raw['CMUN'] = df_M3_raw['CTOT'].str[2:]

    # Add extra digit to CCI AdmBound CTOT - if it contains less than 7 characters
    gdf_AdmBound['CTOT'] = gdf_AdmBound['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Creat for CCI AdmBound column CMUN out of CTOT
    gdf_AdmBound['CMUN'] = gdf_AdmBound['CTOT'].str[2:]

    # Change projection to measure in meters
    gdf_AdmBound = gdf_AdmBound.to_crs(epsg=2062)

    # create new column for surface of municipality in km2
    gdf_AdmBound['Mun_Surface_km2'] = gdf_AdmBound['geometry'].area / 1000**2

    # Merge datasets of M3 and AdmBound into one dataframe
    df_M3 = df_M3_raw.merge(gdf_AdmBound, how = 'right')

    # create new column for relative length of cycleways per 100 km2
    df_M3['M3'] = (df_M3['M3_length'] / df_M3['Mun_Surface_km2']) * 100

    # Replace NaN values with 0 in dataframe
    if FILL_NAN_W_0 is True:
        df_M3['M3'] = df_M3['M3'].fillna(0)

    # filter columns that refer to KPIs
    df_M3 = df_M3[[
        'CMUN', 
        'M3']]

    # exports the dataframe into csv file
    file_name = 'M3_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_M3.to_csv((INTERIM_FOLDER) + export_name, index=False)

    ## M4

    # read datasets into dataframe
    df_M4_raw = pd.read_csv(M4_RAWDATA)
    df_POP21 = pd.read_csv(POP21_INTERIMDATA)

    # DATA HOMOGENIZATION
    # Add extra digit to CCI POP21 CMUN - if it contains less than 5 characters
    df_POP21['CMUN'] = df_POP21['CMUN'].apply(lambda x: '{0:0>5}'.format(x))

    # Add extra digit to CCI M4 CTOT - if it contains less than 7 characters
    df_M4_raw['CTOT'] = df_M4_raw['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Creat for CCI M2 column CMUN out of CTOT
    df_M4_raw['CMUN'] = df_M4_raw['CTOT'].str[2:]

    # Merge datasets of M4 and POP21 into one dataframe
    df_M4 = df_M4_raw.merge(df_POP21, how = 'right')

    # create new column for relative number of bus_stops per 100 inhabitants
    df_M4['M4'] = (df_M4['M4_absolute'] / df_M4['POP21']) * 100

    # Replace NaN values with 0 in dataframe
    if FILL_NAN_W_0 is True:
        df_M4['M4'] = df_M4['M4'].fillna(0)

    # Drop duplicate rows
    df_M4 = df_M4.drop_duplicates(subset=['CMUN'])

    # filter columns that refer to KPIs
    df_M4 = df_M4[[
        'CMUN', 
        'M4']]

    # exports the dataframe into csv file
    file_name = 'M4_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_M4.to_csv((INTERIM_FOLDER) + export_name, index=False)

    ## W2

    # excel files in the path
    file_list = glob(W2_RAW_FOLDER + "/*.xlsx")

    # list of excel files we want to merge data into pandas dataframe.
    excl_list = []

    for file in file_list:  
        df = pd.read_excel(file)
        excl_list.append(df)

    # create a new dataframe to store the merged excel file.
    excl_merged = pd.DataFrame()

    # appends the data into the excl_merged 
    for excl_file in excl_list:
            
        excl_merged = excl_merged.append(
        excl_file, ignore_index=True)

    # filter columns that refer to municipality name and kilograms waste
    df = excl_merged[[
    'Municipio',
    'Envases de vidrio (kgs) recogidos']]

    # create a dictionary
    # key = old name
    # value = new name
    dict = {'Municipio': 'Municipality',
            'Envases de vidrio (kgs) recogidos': 'W2'}
    
    # call rename () method
    df.rename(columns=dict,
            inplace=True)

    # data cleaning of summed up rows per seperate table
    df = df.drop(df.index[df['Municipality'] == "TOTAL"])

    # DATA HOMOGENIALIZATION 
    # Title case to all column names
    df['Municipality'] = df['Municipality'].str.title()

    # Split municipality names into core name and preposition by comma
    df[['MUN_name', 'MUN_preposition']] = df['Municipality'].str.split(',', expand=True)

    # Fill None-value (municipalities without preposition) with empty
    df = pd.DataFrame(df)
    df = df.fillna('')

    # DATA CLEANING
    preposition = ["Es", "La", "El", "Las", "Els", "Los", "A", "O", "Os", "As", "Les", "Sa", "Ses"]

    # Replace preposition before municipality name
    for index, item in enumerate(df['MUN_preposition']):
        try:
            if df['MUN_preposition'][index] == "L'":
                df['Municipality'][index] = df['MUN_preposition'][index] + df['MUN_name'][index]
                df['MUN_preposition'][index] == ''
            elif df['MUN_preposition'][index] in preposition:
                df['Municipality'][index] = df['MUN_preposition'][index]  + df['MUN_name'][index]
                df['MUN_preposition'][index] = ''
            else:
                pass
        except KeyError:
            pass    

    # Replace manually other Municipality names to be in line with administrative dataset
    df['Municipality'].mask(df['Municipality'] == "Fondó De Les Neus, El/Hondón De Las Nieves", "El Fondó De Les Neus/Hondón De Las Nieves", inplace=True)
    df['Municipality'].mask(df['Municipality'] == "Pinós, El/Pinoso", "El Pinós/Pinoso", inplace=True)
    df['Municipality'].mask(df['Municipality'] == "Alqueries, Les/Alquerías Del Niño Perdido", "Les Alqueries/Alquerías Del Niño Perdido", inplace=True)
    df['Municipality'].mask(df['Municipality'] == "Rasillo De Cameros, El", "El Rasillo De Cameros", inplace=True)
    df['Municipality'].mask(df['Municipality'] == "Redal, El", "El Redal", inplace=True)
    df['Municipality'].mask(df['Municipality'] == "Villar De Arnedo, El", "El Villar De Arnedo", inplace=True)


    # filter columns that refer to municipality name and W2-score
    df = df[[
    'Municipality',
    'W2']]

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)
    shp_names = shp_municipality['Municipali']

    # Create new column to append fuzzywuzzy to
    fuzzywuzzy_W2 = pd.DataFrame(columns=['W2_name','W2','Match_name', 'CTOT', 'score', 'fuzzywuzzy_method'])

    # Apply fuzzywuzzy to link municipality names for merging
    n=0

    # try fuzz.token_sort_ratio
    for df_name in df['Municipality']:
        print("Search name : ", df_name, '\n', 62*'-', '\n')
        print(n, "/", len(df.axes[0]))
        
        choices = shp_names
        result = process.extract(df_name, choices, scorer=fuzz.token_sort_ratio)        

        W2 = df['W2'].iloc[n]
        match = result[0][0]
        score = result[0][1]
        index = result[0][2]
        CTOT = shp_municipality['CTOT'].iloc[index]

        if score > 85:
            new_row = {'W2_name':df_name, 
                        'W2':W2,
                        'Match_name':match,                    
                        'CTOT':CTOT,
                        'score':score,
                        'fuzzywuzzy_method': 'token_sort_ratio'}

        # try fuzz.token_set_ratio --> ignores duplicated words
        else:
            result_else = process.extract(df_name, choices, scorer=fuzz.token_set_ratio)

            match_else = result_else[0][0]
            score_else = result_else[0][1]
            index_else = result_else[0][2]
            CTOT_else = shp_municipality['CTOT'].iloc[index_else]

            if score_else > score:
                new_row = {'W2_name':df_name, 
                    'W2':W2,
                    'Match_name':match_else,
                    'CTOT':CTOT_else,
                    'score':score_else,
                    'fuzzywuzzy_method': 'token_set_ratio'}

            #try fuzz.WRatio
            else:
                result_WRatio = process.extract(df_name, choices, scorer=fuzz.WRatio)

                match_WRatio = result_WRatio[0][0]
                score_WRatio = result_WRatio[0][1]
                index_WRatio = result_WRatio[0][2]
                CTOT_WRatio = shp_municipality['CTOT'].iloc[index_WRatio]

                if score_WRatio > score:
                    new_row = {'W2_name':df_name, 
                        'W2':W2,
                        'Match_name':match_WRatio,
                        'score':score_WRatio,
                        'CTOT':CTOT_WRatio,
                        'fuzzywuzzy_method': 'WRatio'}
                
                # If no score above 85, return NAN-value
                else:
                    new_row = {'W2_name':df_name, 
                        'W2':W2,
                        'Match_name':np.NAN,                    
                        'CTOT':np.NAN,
                        'score':np.NAN,
                        'fuzzywuzzy_method': np.NAN}


        fuzzywuzzy_W2 = fuzzywuzzy_W2.append(new_row, ignore_index=True)

        n = n+1

    # filter columns that refer to municipality name and population
    fuzzywuzzy_W2['W2_absolute'] = fuzzywuzzy_W2['W2']

    df = fuzzywuzzy_W2[[
    'CTOT',
    'W2_absolute']]

    # drop duplicate rows
    df = df.drop_duplicates(subset=['CTOT'], keep='first')

    # Get all CTOT codes of municipalities
    shp_CTOT = shp_municipality['CTOT']

    # Merge W2 dataset with all CTOT municipality dataset
    df_merge = df.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0 (if municipality does not have punto limpio)
    df_merge['W2_absolute'] = df_merge['W2_absolute'].fillna(0)

    df_W2 = df_merge[['CTOT', 'W2_absolute']]

    # Normalize outcome to population of municipality

    # read datasets into dataframe
    df_POP21 = pd.read_csv(POP21_INTERIMDATA)

    # Add extra digit to CCI POP21 CMUN - if it contains less than 5 characters
    df_POP21['CMUN'] = df_POP21['CMUN'].apply(lambda x: '{0:0>5}'.format(x))

    # Add extra digit to CCI W2 CTOT - if it contains less than 7 characters
    df_W2['CTOT'] = df_W2['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

    # Create for CCI W2 column CMUN out of CTOT
    df_W2['CMUN'] = df_W2['CTOT'].str[2:]

    # Merge datasets of W2 and POP21 into one dataframe
    df_merge_POP21 = pd.concat([df_W2, df_POP21], axis=1)

    # create new column for relative number of waste per inhabitant
    df_merge_POP21['W2'] = (df_merge_POP21['W2_absolute'] / df_merge_POP21['POP21'])

    # filter columns that refer to KPIs
    df = df_merge_POP21[[
        'CTOT', 
        'W2']]

    # exports the dataframe into csv file with
    file_name = 'EXTRA_W2_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df.to_csv(INTERIM_FOLDER + export_name, index=False)

    ## W3

    # excel files in the path
    file_list = glob(W3_RAW_FOLDER + "/*.xlsx")

    # list of excel files we want to merge data into pandas dataframe.
    excl_list = []
    print(excl_list)

    for file in file_list:  
        # read file
        df = pd.read_excel(file)
        
        # replace column header to first row 
        df=pd.DataFrame(df.columns.values[None,:],columns=df.columns).\
        append(df).\
        reset_index(drop=True)

        # add a title to first column
        df.set_axis(["Info"], axis=1,inplace=True)

        # select odd and even rows to seperate punto limpio from adress information
        df_odd = df.iloc[::2]
        df_even = df.iloc[1::2]
        df['puntos_limpios'] = df_odd
        df['information'] = df_even

        # remove NaN values from columns
        df = df.apply(lambda x: pd.Series(x.dropna().values))
        df = df.fillna('')

        # drop first column
        df = df.drop(['Info'], axis=1)
        df = df.fillna('')
        df_length = int(df.shape[0])
        df_halflength = int(df_length * 0.5)
        df = df.drop(range(df_halflength, df_length))
        
        #append datasets
        excl_list.append(df)

    # create a new dataframe to store the merged excel file.
    excl_merged = pd.DataFrame()
    
    for excl_file in excl_list:
        
        # appends the data into the excl_merged
        # dataframe.
        excl_merged = excl_merged.append(
        excl_file, ignore_index=True)

    # title case-format to names of puntos limpios
    excl_merged['puntos_limpios'].str.title()
    df = excl_merged

    # Merge columns
    df['text'] = df['puntos_limpios'] + " " + df['information']

    # DATA CLEANING
    # exports the dataframe into csv file with
    W3_location = INTERIM_FOLDER + 'W3_temporaldata.csv'
    df.to_csv(W3_location, index=False)

 
    nltk.download('punkt')
    nltk.download('stopwords')

    STOP_WORDS = stopwords.words()
    EMOJI_PATTERN = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)


    def cleaning(text):
        """
        Convert to lowercase.
        Rremove URL links, special characters and punctuation.
        Tokenize and remove stop words.
        """
        text = text.lower()
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('[’“”…]', '', text)
        text = re.sub(r'[0-9]+', '', text) # remove numbers

        text = EMOJI_PATTERN.sub(r'', text)

        # removing the stop-words
        text_tokens = word_tokenize(text)
        tokens_without_sw = [
            word for word in text_tokens if not word in STOP_WORDS]
        filtered_sentence = (" ").join(tokens_without_sw)
        text = filtered_sentence

        return text
        return word_list


    if __name__ == "__main__":
        max_rows = None  # 'None' to read whole file
        input_file = W3_location
        df = pd.read_csv(input_file,
                        delimiter = ',',
                        nrows = max_rows,
                        engine = "python")

        dt = df['text'].apply(cleaning)

        word_count = Counter(" ".join(dt).split()).most_common(20)
        word_frequency = pd.DataFrame(word_count, columns = ['Word', 'Frequency'])
        print(word_frequency)

    df = pd.DataFrame(dt)
    df['text'] = df

    # Remove temporal file
    os.remove(W3_location)

    # DATA CLEANING
    # Create list with most mentioned words
    word_list = pd.DataFrame(word_count, columns = ['Word', 'Frequency'])
    word_list = word_list['Word'].tolist()

    # Create extra list for error words
    extra_list = ['mancomunidad', 'cañetemunicipio', 'nº', 'área aportación']

    # Create list with exceptions in most mentioned list
    exception = ['san', 'sant', 'santa']

    word_list = word_list + extra_list

    # Remove exceptions from list
    for word in word_list:
        if word in exception:
            word_list.remove(word)

    for word in word_list:
        print(word)
        df['text'] = df['text'].str.replace(word, '')

    # read Shapefiles for foundation of municipality names
    shp_municipality = gpd.read_file(ADMBOUND_INTERIMDATA)
    shp_names = shp_municipality['Municipali']

    # Create new column to append fuzzywuzzy to
    fuzzywuzzy_W3 = pd.DataFrame(columns=['W3_name','Match_name', 'CTOT', 'score', 'fuzzywuzzy_method'])

    # Apply fuzzywuzzy to link municipality names for merging
    n=0

    # try fuzz.token_set_ratio --> ignores duplicated words
    for df_name in df['text']:
        print("Search name : ", df_name, '\n', 62*'-', '\n')
        print(n, "/", len(df.axes[0]))
        
        choices = shp_names
        result = process.extract(df_name, choices, scorer=fuzz.token_set_ratio)        

        match = result[0][0]
        score = result[0][1]
        index = result[0][2]
        CTOT = shp_municipality['CTOT'].iloc[index]

        if score > 85:
            new_row = {'W3_name':df_name,                     
                        'Match_name':match,                    
                        'CTOT':CTOT,
                        'score':score,
                        'fuzzywuzzy_method': 'token_set_ratio'}

        # try fuzz.token_sort_ratio
        else:
            result_else = process.extract(df_name, choices, scorer=fuzz.token_sort_ratio)

            match_else = result_else[0][0]
            score_else = result_else[0][1]
            index_else = result_else[0][2]
            CTOT_else = shp_municipality['CTOT'].iloc[index_else]

            if score_else > score:
                new_row = {'W3_name':df_name,                 
                    'Match_name':match_else,
                    'CTOT':CTOT_else,
                    'score':score_else,
                    'fuzzywuzzy_method': 'token_sort_ratio'}

            #try fuzz.WRatio
            else:
                result_WRatio = process.extract(df_name, choices, scorer=fuzz.WRatio)

                match_WRatio = result_WRatio[0][0]
                score_WRatio = result_WRatio[0][1]
                index_WRatio = result_WRatio[0][2]
                CTOT_WRatio = shp_municipality['CTOT'].iloc[index_WRatio]

                if score_WRatio > score:
                    new_row = {'W3_name':df_name,                     
                        'Match_name':match_WRatio,
                        'score':score_WRatio,
                        'CTOT':CTOT_WRatio,
                        'fuzzywuzzy_method': 'WRatio'}
                
                # If no score above 85, return NAN-value
                else:
                    new_row = {'W3_name':df_name,                     
                        'Match_name':np.NAN,                    
                        'CTOT':np.NAN,
                        'score':np.NAN,
                        'fuzzywuzzy_method': np.NAN}


        fuzzywuzzy_W3 = fuzzywuzzy_W3.append(new_row, ignore_index=True)

        n = n+1

    # Add a 1, if municipality has a punto limpio
    fuzzywuzzy_W3['W3'] = 1

    df = fuzzywuzzy_W3[[
    'CTOT',
    'W3']]

    # drop duplicate rows
    df = df.drop_duplicates(subset=['CTOT'], keep='first')

    # Get all CTOT codes of municipalities
    shp_CTOT = shp_municipality['CTOT']

    # Merge W3 dataset with all CTOT municipality dataset
    df_merge = df.merge(shp_municipality, how = 'right')

    # filter out NaN-values and fill with 0 (if municipality does not have punto limpio)
    df_merge['W3'] = df_merge['W3'].fillna(0)

    df = df_merge[[
    'CTOT',
    'W3']]

    # exports the dataframe into csv file with
    file_name = 'W3_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df.to_csv(INTERIM_FOLDER + export_name, index=False)

    # PART 3 - SOCIODEMOGRAPHIC & ECONOMIC DATA

    ## Income

    # read the raw csv file from the url
    df_income = pd.read_excel(INCOME_RAWDATA)

    # Data cleaning
    df_income.dropna(inplace=True)
    df_income['CMUN'] = df_income['Atlas de distribución de renta de los hogares'].str.split(' ').str[0]
    df_income = df_income[[len(i) <= 5 for i in df_income['CMUN']]]

    # Drop first row
    df_income = df_income.tail(-1)

    # create a dictionary
    dict = {
        'Unnamed: 2': 'INCOME_PER_CAPITA_2020',
        'Unnamed: 7': 'INCOME_PER_HOUSEHOLD_2020',}
    
    # call rename () method
    df_income.rename(columns=dict,
            inplace=True)

    # filter columns
    df_income = df_income[[
        'CMUN', 
        'INCOME_PER_CAPITA_2020', 
        'INCOME_PER_HOUSEHOLD_2020'
        ]]

    # exports the dataframe into csv file with
    file_name = 'INCOME_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_income.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Demographic

    # read the raw excel file from the url
    df_demographic = pd.read_excel(DEMOGRAPHIC_RAWDATA)

    df_demographic.dropna(inplace=True)
    df_demographic['CMUN'] = df_demographic['Atlas de distribución de renta de los hogares'].str.split(' ').str[0]
    df_demographic = df_demographic[[len(i) <= 5 for i in df_demographic['CMUN']]]

    # Drop first row
    df_demographic = df_demographic.tail(-1)

    # create a dictionary
    dict = {
        'Unnamed: 1': 'AGE_AVERAGE_2020',
        'Unnamed: 7': 'AGE_PERC_POP_BELOW_18_2020',
        'Unnamed: 13': 'AGE_PERC_POP_ABOVE_65_2020',
        'Unnamed: 19': 'HOUSING_AVERAGE_HOUSEHOLD_SIZE_2020',
        'Unnamed: 25': 'HOUSING_PERC_SINGLEPERSON_HOUSEHOLD_2020',
        'Unnamed: 31': 'POPULATION_2020',
        'Unnamed: 37': 'NATIONALITY_PERC_SPANISH_2020'}

    # call rename () method
    df_demographic.rename(columns=dict,
            inplace=True)

    # Read spatial geodataframe
    gdf_AdmBound = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Change projection to measure in meters
    gdf_AdmBound = gdf_AdmBound.to_crs(epsg=2062)

    # Creat column CMUN out of CTOT
    gdf_AdmBound['CMUN'] = gdf_AdmBound['CTOT'].str[2:]

    # Calculate surface in km2 per municipality
    gdf_AdmBound['SURFACE_MUN'] = gdf_AdmBound['geometry'].area / 1000**2
    df_AdmBound = pd.DataFrame(gdf_AdmBound, columns=['CMUN','SURFACE_MUN'])

    # Merge datasets on CMUN
    df_demographic = pd.merge(df_demographic, df_AdmBound, on='CMUN')

    # Calculate population density per municipality
    df_demographic['POPULATION_DENSITY_KM2_2020'] = df_demographic['POPULATION_2020'] / df_demographic['SURFACE_MUN']

    df_demographic['NATIONALITY_PERC_NONSPANISH_2020'] = 100 - df_demographic['NATIONALITY_PERC_SPANISH_2020']

    # filter columns
    df_demographic = df_demographic[[
        'CMUN',
        'POPULATION_2020',     
        'POPULATION_DENSITY_KM2_2020',
        'AGE_AVERAGE_2020', 
        'AGE_PERC_POP_BELOW_18_2020',
        'AGE_PERC_POP_ABOVE_65_2020',
        'HOUSING_AVERAGE_HOUSEHOLD_SIZE_2020',
        'HOUSING_PERC_SINGLEPERSON_HOUSEHOLD_2020',    
        'NATIONALITY_PERC_SPANISH_2020',
        'NATIONALITY_PERC_NONSPANISH_2020'
        ]]

    # exports the dataframe into csv file with
    file_name = 'DEMOGRAPHIC_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_demographic.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## GINI

    # read the raw excel file from the url
    df_gini = pd.read_excel(GINI_RAWDATA)

    # Clean datasets
    df_gini.dropna(inplace=True)
    df_gini['CMUN'] = df_gini['Atlas de distribución de renta de los hogares'].str.split(' ').str[0]
    df_gini = df_gini[[len(i) <= 5 for i in df_gini['CMUN']]]

    # Drop first row
    df_gini = df_gini.tail(-1)

    # create a dictionary
    dict = {
        'Unnamed: 1': 'WEALTH_GINI_2020'}

    # call rename () method
    df_gini.rename(columns=dict,
            inplace=True)

    df_gini['WEALTH_GINI_2020'] = df_gini['WEALTH_GINI_2020'].astype('str')
    df_gini.loc[df_gini['WEALTH_GINI_2020'].str.startswith('.', na=False), 'WEALTH_GINI_2020'] = np.nan
    df_gini.loc[df_gini['WEALTH_GINI_2020'].str.startswith(' ', na=False), 'WEALTH_GINI_2020'] = np.nan

    df_gini['WEALTH_GINI_2020'] = df_gini['WEALTH_GINI_2020'].astype('float')

    # filter columns
    df_gini = df_gini[[
        'CMUN',     
        'WEALTH_GINI_2020'
        ]]

    df_gini

    # exports the dataframe into csv file with
    file_name = 'GINI_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_gini.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Demographic growth

    # read the raw excel file from the url
    df_dem_growth = pd.read_excel(DEM_GROWTH_RAWDATA)

    # Clean datasets
    df_dem_growth.drop(df_dem_growth.index[:5], inplace=True)
    df_dem_growth['CMUN'] = df_dem_growth['Fenómenos demográficos.Resumen municipal y series.'].str.split(' ').str[0]

    # create a dictionary
    dict = {
        'Unnamed: 5': 'NATURAL_POP_GROWTH_2020'}

    # call rename () method
    df_dem_growth.rename(columns=dict,
            inplace=True)

    # Calculate percentage growth
    df_dem_growth = pd.merge(df_dem_growth, df_demographic, on='CMUN')
    df_dem_growth['POPULATION_PERC_NATURAL_GROWTH_2020'] = (df_dem_growth['NATURAL_POP_GROWTH_2020'] / df_dem_growth['POPULATION_2020']) * 100

    # filter columns
    df_dem_growth = df_dem_growth[[
        'CMUN',     
        'POPULATION_PERC_NATURAL_GROWTH_2020'
        ]]

    # exports the dataframe into csv file with
    file_name = 'DEM_GROWTH_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_dem_growth.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Economy - total companies by sector

    # read the raw excel file from the url
    df_economy_company = pd.read_excel(ECONOMIC_COMPANY_RAWDATA)

    # Clean datasets
    # Remove first 8 and last 8 rows
    df_economy_company.drop(df_economy_company.index[:8], inplace=True)
    df_economy_company.drop(df_economy_company.tail(8).index, inplace = True)

    # Strip white spaces from first columm
    df_economy_company['CMUN'] = df_economy_company['Resultados municipales   '].str.lstrip()

    # Obtain CMUN
    df_economy_company['CMUN'] = df_economy_company['CMUN'].str.split(' ').str[0]
    df_economy_company = df_economy_company[[len(i) >= 5 for i in df_economy_company['CMUN']]]

    # Drop first row
    df_economy_company = df_economy_company.tail(-1)

    # create a dictionary
    dict = {
        'Unnamed: 1': 'TOTAL_COMPANIES_2020'}

    # call rename () method
    df_economy_company.rename(columns=dict,
            inplace=True)

    # Merge dataset with population data
    df_economy_company = pd.merge(df_economy_company, df_demographic, on='CMUN')

    # Fill missing data with 0
    df_economy_company["TOTAL_COMPANIES_2020"] = df_economy_company["TOTAL_COMPANIES_2020"].astype('str')
    df_economy_company.loc[df_economy_company["TOTAL_COMPANIES_2020"].str.startswith(' ', na=False), "TOTAL_COMPANIES_2020"] = ""
    df_economy_company.loc[df_economy_company["TOTAL_COMPANIES_2020"].str.startswith('.', na=False), "TOTAL_COMPANIES_2020"] = 0
    df_economy_company["TOTAL_COMPANIES_2020"].fillna(0)

    df_economy_company["POPULATION_2020"] = df_demographic["POPULATION_2020"].astype('str')
    df_economy_company.loc[df_economy_company["POPULATION_2020"].str.startswith(' ', na=False), "TOTAL_COMPANIES_2020"] = ""
    df_economy_company["POPULATION_2020"].fillna(0)

    # Calculate companies per capita in municipality
    df_economy_company["ECONOMY_COMPANIES_PER_CAPITA_2020"] = (
        pd.to_numeric(df_economy_company["TOTAL_COMPANIES_2020"], errors='coerce') 
        / pd.to_numeric(df_economy_company["POPULATION_2020"], errors='coerce')
    )

    # filter columns
    df_economy_company = df_economy_company[[
        'CMUN',
        'ECONOMY_COMPANIES_PER_CAPITA_2020'
        ]]

    # exports the dataframe into csv file with
    file_name = 'ECONOMIC_COMPANY_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_economy_company.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Outstanding debt of Municipality

    # read the raw excel file
    df_debt_MUN = pd.read_excel(DEBT_MUNICIPALITY_RAWDATA)

    # Clean datasets
    # Remove first 9 and last 2 rows
    df_debt_MUN.drop(df_debt_MUN.index[:10], inplace=True)
    df_debt_MUN.drop(df_debt_MUN.tail(2).index, inplace = True)
    df_debt_MUN['CMUN'] = df_debt_MUN['Unnamed: 3'] + df_debt_MUN['Unnamed: 5']

    # create a dictionary
    dict = {
        'Unnamed: 7': 'DEBT_MUNICIPALITY_2021'}

    # call rename () method
    df_debt_MUN.rename(columns=dict,
            inplace=True)

    # Calculate debt per capita
    df_debt_MUN = pd.merge(df_debt_MUN, df_demographic, on='CMUN')
    df_debt_MUN['DEBT_MUNICIPALITY_PER_CAPITA_2021'] = df_debt_MUN['DEBT_MUNICIPALITY_2021'] / df_debt_MUN['POPULATION_2020']

    # filter columns
    df_debt_MUN = df_debt_MUN[[
        'CMUN', 
        'DEBT_MUNICIPALITY_PER_CAPITA_2021'
        ]]

    # exports the dataframe into csv file with
    file_name = 'DEBT_MUNICIPALITY_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_debt_MUN.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Population (per gender)

    # read the raw excel file from the url
    df_pop_gender = pd.read_excel(POP_GENDER_RAWDATA)

    # Clean datasets
    # Remove first 7 and last 6 rows
    df_pop_gender.drop(df_pop_gender.index[:8], inplace=True)
    df_pop_gender.drop(df_pop_gender.tail(6).index, inplace = True)

    # Obtain CMUN
    df_pop_gender['CMUN'] = df_pop_gender['Población'].str.split(' ').str[0]
    df_pop_gender = df_pop_gender[[len(i) >= 5 for i in df_pop_gender['CMUN']]]

    # Calculate percentage of male and female population
    df_pop_gender['GENDER_PERC_POP_MALE_2020'] = df_pop_gender['Unnamed: 2'] / df_pop_gender['Unnamed: 1'] * 100
    df_pop_gender['GENDER_PERC_POP_FEMALE_2020'] = df_pop_gender['Unnamed: 3'] / df_pop_gender['Unnamed: 1'] * 100

    # filter columns
    df_pop_gender = df_pop_gender[[
        'CMUN', 
        'GENDER_PERC_POP_MALE_2020', 
        'GENDER_PERC_POP_FEMALE_2020'
    ]]

    # exports the dataframe into csv file with
    file_name = 'POP_GENDER_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_pop_gender.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Residential buildings

    # read the raw excel file from the url
    df_residential_buildings = pd.read_excel(RESIDENTIAL_BUILDINGS_RAWDATA)

    # Clean datasets
    # Remove first 8 and last 8 rows
    df_residential_buildings.drop(df_residential_buildings.index[:7], inplace=True)
    df_residential_buildings.drop(df_residential_buildings.tail(6).index, inplace = True)

    # Obtain CMUN
    df_residential_buildings['CMUN'] = df_residential_buildings['Censos de Población y Viviendas 2011. Viviendas. Resultados Municipales. Principales resultados'].str.split(' ').str[0]
    df_residential_buildings = df_residential_buildings[[len(i) >= 5 for i in df_residential_buildings['CMUN']]]

    # Calculate total resident buildings
    df_residential_buildings['RESIDENT_BUILDINGS'] = df_residential_buildings['Unnamed: 1'] + df_residential_buildings['Unnamed: 2']

    # Calculate resident buildings per capita
    df_residential_buildings = pd.merge(df_residential_buildings, df_demographic, on='CMUN')
    df_residential_buildings['HOUSING_RESIDENT_BUILDINGS_PER_CAPITA_2011'] = df_residential_buildings['RESIDENT_BUILDINGS'] / df_residential_buildings['POPULATION_2020']

    # filter columns
    df_residential_buildings = df_residential_buildings[[
        'CMUN',  
        'HOUSING_RESIDENT_BUILDINGS_PER_CAPITA_2011'
    ]]

    # exports the dataframe into csv file with
    file_name = 'RESIDENT_BUILDINGS_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_residential_buildings.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Tourist houses

    # read the raw excel file from the url
    df_tourist_houses = pd.read_excel(TOURIST_HOUSES_RAWDATA)

    # Clean datasets
    # Remove first 7 and last 6 rows
    df_tourist_houses.drop(df_tourist_houses.index[:8], inplace=True)
    df_tourist_houses.drop(df_tourist_houses.tail(6).index, inplace = True)

    # Obtain CMUN
    df_tourist_houses['CMUN'] = df_tourist_houses['Viviendas turisticas en España'].str.split(' ').str[0]
    df_tourist_houses = df_tourist_houses[[len(i) >= 5 for i in df_tourist_houses['CMUN']]]

    # Calculate tourist houses per capita
    df_tourist_houses = pd.merge(df_tourist_houses, df_demographic, on='CMUN')
    df_tourist_houses['TOURISM_HOUSES_PER_CAPITA_2022'] = df_tourist_houses['Unnamed: 1'] / df_tourist_houses['POPULATION_2020']

    # filter columns
    df_tourist_houses = df_tourist_houses[[
        'CMUN',  
        'TOURISM_HOUSES_PER_CAPITA_2022'
    ]]

    # exports the dataframe into csv file with
    file_name = 'TOURIST_HOUSES_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_tourist_houses.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Unemployment benefits

    # read the raw excel file from the url
    df_unemployment_benefits = pd.read_excel(UNEMPLOYMENT_BENEFITS_RAWDATA)

    # Clean datasets
    # Remove first 7 and last 6 rows
    df_unemployment_benefits.drop(df_unemployment_benefits.index[:8], inplace=True)
    df_unemployment_benefits.drop(df_unemployment_benefits.tail(7).index, inplace = True)

    # Obtain CMUN
    df_unemployment_benefits['CMUN'] = df_unemployment_benefits['Atlas de distribución de renta de los hogares'].str.split(' ').str[0]
    df_unemployment_benefits = df_unemployment_benefits[[len(i) == 5 for i in df_unemployment_benefits['CMUN']]]

    # Calculate unemployment benefits as percentage of total average salary
    df_unemployment_benefits["INCOME_PERC_UNEMPLOYMENT_BENEFITS_OF_AVERAGE_SALARY_2020"] = (
        pd.to_numeric(df_unemployment_benefits["Unnamed: 13"], errors='coerce') 
        / pd.to_numeric(df_unemployment_benefits["Unnamed: 1"], errors='coerce') * 100
    )

    # filter columns
    df_unemployment_benefits = df_unemployment_benefits[[
        'CMUN',  
        'INCOME_PERC_UNEMPLOYMENT_BENEFITS_OF_AVERAGE_SALARY_2020'
    ]]

    # exports the dataframe into csv file with
    file_name = 'UNEMPLOYMENT_BENEFITS_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_unemployment_benefits.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    ## Agriculture

    # read the raw excel file from the url
    df_agriculture = pd.read_excel(AGRICULTURE_RAWDATA)

    # Clean datasets
    # Remove first 7 and last 8 rows
    df_agriculture.drop(df_agriculture.index[:8], inplace=True)
    df_agriculture.drop(df_agriculture.tail(8).index, inplace = True)

    # Obtain CMUN
    df_agriculture['CMUN'] = df_agriculture['Resultados estructurales nacionales, por comunidades autónomas, provincias, comarcas y municipios'].str.split(' ').str[0]
    df_agriculture = df_agriculture[[len(i) == 5 for i in df_agriculture['CMUN']]]

    # Replace '..' with 0
    df_agriculture = df_agriculture.replace('..', 0)

    # Read spatial geodataframe
    gdf_AdmBound = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Change projection to measure in meters
    gdf_AdmBound = gdf_AdmBound.to_crs(epsg=2062)

    # Creat column CMUN out of CTOT
    gdf_AdmBound['CMUN'] = gdf_AdmBound['CTOT'].str[2:]

    # Calculate surface in km2 per municipality
    gdf_AdmBound['SURFACE_MUN'] = gdf_AdmBound['geometry'].area / 1000**2
    df_AdmBound = pd.DataFrame(gdf_AdmBound, columns=['CMUN','SURFACE_MUN'])

    # Merge datasets on CMUN
    df_agriculture = pd.merge(df_agriculture, df_AdmBound, on='CMUN')

    # Calculate population density per municipality
    df_agriculture['AGRI_LIVESTOCKUNITS_DENSITY_KM2_2020'] = df_agriculture['Unnamed: 3'] / df_agriculture['SURFACE_MUN']
    df_agriculture['AGRI_CATTLEFARMS_DENSITY_KM2_2020'] = df_agriculture['Unnamed: 1'] / df_agriculture['SURFACE_MUN']

    # filter columns
    df_agriculture = df_agriculture[[
        'CMUN',  
        'AGRI_LIVESTOCKUNITS_DENSITY_KM2_2020',
        'AGRI_CATTLEFARMS_DENSITY_KM2_2020'   
    ]]

    # exports the dataframe into csv file with
    file_name = 'AGRICULTURE_interimdata'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INTERIMDATA is True:
        df_agriculture.to_csv(INTERIM_DEMOGRAPHIC_FOLDER + export_name, index=False)

    print("Data Management is finished")
    print("##################################")
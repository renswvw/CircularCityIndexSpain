from pathlib import Path
import geopandas as gpd
import pandas as pd
import os
from glob import glob
import numpy as np

def DataMerge():
    # papermill parameters cell
    OUTPUT_WARNINGS = False
    SAVE_MERGEDDATA = True

    if OUTPUT_WARNINGS is False:
        import warnings

        warnings.filterwarnings("ignore")

    ### Data
    # Data folders
    INTERIM_FOLDER = 'data/interim/'
    ADMBOUND_INTERIM_FOLDER = 'data/interim/AdmBound_interimdata/'
    SPATIALIZED_FOLDER = "data/interim/Spatialization_interimdata/"
    DEMOGRAPHIC_INTERIM_FOLDER = "data/interim/demographic_interimdata/"
    MERGED_DEMOGRAPHIC_INTERIM_FOLDER = "data/interim/demographic_interimdata/merged_demographic_interimdata/"

    # create folder data/raw/ if not exists
    Path(SPATIALIZED_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(MERGED_DEMOGRAPHIC_INTERIM_FOLDER).mkdir(parents=True, exist_ok=True)

    # Datasets 
    ADMBOUND_INTERIMDATA = ADMBOUND_INTERIM_FOLDER + 'AdmBound_interimdata.gpkg'

    # PART 1 - KPI INDEX DATA
    ## Data Merge
    # read interim dataset of Administrative Boundaries into geodataframe
    gdf = gpd.read_file(ADMBOUND_INTERIMDATA)

    # Creat column CMUN out of CTOT
    gdf['CMUN'] = gdf['CTOT'].str[2:]

    # Create list with KPI names
    KPI_list = [
        'D1',
        'D2',
        'D3',
        'D4',
        'ECR1',
        'ECR2',
        'ECR4',
        'ECR5',
        'M1',
        'M2',
        'M3',
        'M4',
        'W2',
        'W3',
        'POP21'
    ]

    # DATA MANAGEMENT --> SCRIPT MOET WORDEN AFGEMAAKT, KLOPT NU NIET
    dataframe_list = []

    for file in glob(INTERIM_FOLDER + '*.csv'):
        # import interim datasets of CCIs
        file_name = os.path.basename(file)
        file_name = file_name.split('_', 1)[0]
        df = pd.read_csv(file)

        # DATA HOMOGENIZATION
        # Add extra digit to dataset['CMUN'] - if it contains less than 5 characters
        if 'CMUN' in df.columns:
            df["CMUN"] = df["CMUN"].apply(lambda x: '{0:0>5}'.format(x))
        
        # Add extra digit to dataset['CTOT'] - if it contains less than 7 characters
        elif 'CTOT' in df.columns :
                    df['CTOT'] = df['CTOT'].apply(lambda x: '{0:0>7}'.format(x))
        else:
            pass

        globals()[f"CCI_{file_name}"] = df
        dataframe_list.append(globals()[f"CCI_{file_name}"])

    # DATA MERGE
    # Merge GeoDataFrame with seperate DataFrames per KPI
    gdf_temporal1 = gpd.GeoDataFrame(CCI_D1.merge(gdf, how = 'right'))
    gdf_temporal2 = gpd.GeoDataFrame(CCI_D2.merge(gdf_temporal1, how = 'right'))
    gdf_temporal3 = gpd.GeoDataFrame(CCI_D3.merge(gdf_temporal2, how = 'right'))
    gdf_temporal4 = gpd.GeoDataFrame(CCI_D4.merge(gdf_temporal3, how = 'right'))
    gdf_temporal5 = gpd.GeoDataFrame(CCI_ECR1.merge(gdf_temporal4, how = 'right'))
    gdf_temporal6 = gpd.GeoDataFrame(CCI_ECR2.merge(gdf_temporal5, how = 'right'))
    gdf_temporal7 = gpd.GeoDataFrame(CCI_ECR4.merge(gdf_temporal6, how = 'right'))
    gdf_temporal8 = gpd.GeoDataFrame(CCI_ECR5.merge(gdf_temporal7, how = 'right'))
    gdf_temporal9 = gpd.GeoDataFrame(CCI_M1.merge(gdf_temporal8, how = 'right'))
    gdf_temporal10 = gpd.GeoDataFrame(CCI_M2.merge(gdf_temporal9, how = 'right'))
    gdf_temporal11 = gpd.GeoDataFrame(CCI_M3.merge(gdf_temporal10, how = 'right'))
    gdf_temporal12 = gpd.GeoDataFrame(CCI_M4.merge(gdf_temporal11, how = 'right'))
    gdf_temporal13 = gpd.GeoDataFrame(CCI_W2.merge(gdf_temporal12, how = 'right'))
    gdf_temporal14 = gpd.GeoDataFrame(CCI_W3.merge(gdf_temporal13, how = 'right'))
    gdf_temporal15 = gpd.GeoDataFrame(CCI_POP21.merge(gdf_temporal14, how = 'right'))
    gdf_temporal15

    # filter columns that refer to KPIs
    gdf_master = gdf_temporal15[[
        'CTOT', 
        'CMUN', 
        'Municipality',
        'geometry', 
        'D1',
        'D2',
        'D3',
        'D4', 
        'ECR1', 
        'ECR2',
        'ECR4',
        'ECR5', 
        'M1',
        'M2',
        'M3',
        'M4',
        'W2',
        'W3',
        'POP21']]

    # Drop duplicate values (contain same values)
    gdf_master = gdf_master.drop_duplicates(subset=['CTOT'])
    gdf_master["W2"] = gdf_master["W2"] / 1000

    if SAVE_MERGEDDATA is True:
        gdf_master.describe().to_csv("data/processed/CCI/Spain/03_index/descriptive_unprocessed_CCI_KPIs.csv", index=True)

    gdf_master.describe()

    ## Export files
    # exports the geodataframe into GeoPackage file
    file_name = 'CCI_spatialization_interimdata'
    data_format = '.gpkg'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        gdf_master.to_file(SPATIALIZED_FOLDER + export_name, driver='GPKG') 

    # exports the geodataframe into Shapefile
    file_name = 'CCI_spatialization_interimdata'
    data_format = '.shp'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        gdf_master.to_file(SPATIALIZED_FOLDER + export_name) 

    # export the geodataframe into csv file
    file_name = 'CCI_spatialization_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        gdf_master.to_csv(SPATIALIZED_FOLDER + export_name, index=False)

    # PART 2 - DEMOGRAPHIC DATA
    df_base = pd.DataFrame(gdf)

    # DATA MANAGEMENT --> SCRIPT MOET WORDEN AFGEMAAKT, KLOPT NU NIET
    dataframe_list = []
    for file in glob(DEMOGRAPHIC_INTERIM_FOLDER + '*.csv'):
        # import interim datasets of CCIs
        df = pd.read_csv(file)

        # DATA HOMOGENIZATION
        # Add extra digit to dataset['CMUN'] - if it contains less than 5 characters
        if 'CMUN' in df.columns:
            df["CMUN"] = df["CMUN"].apply(lambda x: '{0:0>5}'.format(x))
        
        # DATA MERGE
        df_base = pd.merge(df_base, df, on='CMUN')

    # filter columns
    df_demographic_total = df_base[[
        "CTOT", 
        'POPULATION_2020',
        "POPULATION_DENSITY_KM2_2020",
        "POPULATION_PERC_NATURAL_GROWTH_2020",
        #"GENDER_PERC_POP_MALE_2020",
        "GENDER_PERC_POP_FEMALE_2020",
        "AGE_AVERAGE_2020",
        "AGE_PERC_POP_BELOW_18_2020",
        "AGE_PERC_POP_ABOVE_65_2020",   
        #"NATIONALITY_PERC_SPANISH_2020",
        "NATIONALITY_PERC_NONSPANISH_2020",
        "HOUSING_AVERAGE_HOUSEHOLD_SIZE_2020",
        "HOUSING_PERC_SINGLEPERSON_HOUSEHOLD_2020",
        'HOUSING_RESIDENT_BUILDINGS_PER_CAPITA_2011',
        "INCOME_PER_CAPITA_2020",
        "INCOME_PER_HOUSEHOLD_2020",
        'INCOME_PERC_UNEMPLOYMENT_BENEFITS_OF_AVERAGE_SALARY_2020',
        "WEALTH_GINI_2020",
        "DEBT_MUNICIPALITY_PER_CAPITA_2021",
        "ECONOMY_COMPANIES_PER_CAPITA_2020",
        'AGRI_LIVESTOCKUNITS_DENSITY_KM2_2020',
        'AGRI_CATTLEFARMS_DENSITY_KM2_2020',    
        'TOURISM_HOUSES_PER_CAPITA_2022',
        "geometry"
    ]]

    # Drop duplicate values (contain same values)
    df_demographic_total = df_demographic_total.drop_duplicates(subset=['CTOT'])
    df_demographic_total

    # Replace dots with empty
    col_list = ["INCOME_PER_CAPITA_2020", "INCOME_PER_HOUSEHOLD_2020", "WEALTH_GINI_2020"]

    for col in col_list:
        print(col)
        df_demographic_total[col] = df_demographic_total[col].astype('str')
        df_demographic_total.loc[df_demographic_total[col].str.startswith('.', na=False), col] = ' '
        df_demographic_total.loc[df_demographic_total[col].str.startswith(' ', na=False), col] = np.nan

        # Fill empty cells with NAN
        df_demographic_total[col] = df_demographic_total[col].astype('float')

    gdf_demographic_total = gpd.GeoDataFrame(df_demographic_total)

    ## Export files
    # exports the geodataframe into GeoPackage file
    file_name = 'Spatial_demographic_interimdata'
    data_format = '.gpkg'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        gdf_demographic_total.to_file(MERGED_DEMOGRAPHIC_INTERIM_FOLDER + export_name, driver='GPKG') 

    # exports the geodataframe into Shapefile
    file_name = 'Spatial_demographic_interimdata'
    data_format = '.shp'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        gdf_demographic_total.to_file(MERGED_DEMOGRAPHIC_INTERIM_FOLDER + export_name) 

    # export the geodataframe into csv file
    file_name = 'Spatial_demographic_interimdata'
    data_format = '.csv'
    export_name = file_name + data_format
    if SAVE_MERGEDDATA is True:
        df_demographic_total.to_csv(MERGED_DEMOGRAPHIC_INTERIM_FOLDER + export_name, index=False)

    print("Data Merge and Geocoding is finished")
    print("##################################")
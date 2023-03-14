# 03. Index Weighting & Formula
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt

def IndexMaker():
    # papermill parameters cell
    OUTPUT_WARNINGS = False
    SAVE_FIGS = True
    SAVE_INDEX = True
    ALTERNATIVE_KPI_WEIGHT = True # if False: original weightgs, but maximum value in thesis could be 0.76 instead of 1.0
    ALTERNATIVE_AREA_COMPUTATION = True # if False: original area computation equations are used (W2 is strange defined)

    if OUTPUT_WARNINGS is False:
        import warnings

        warnings.filterwarnings("ignore")

    # create list with KPIs
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
        'W3'
    ]

    ### Data
    # Data folders - input
    INTERIM_FOLDER = 'data/interim/'
    SPATIALIZED_FOLDER = "data/interim/Spatialization_interimdata/"
    INDEX_FOLDER = "data/interim/Index_interimdata/"

    # Datasets 
    SPATIALIZED_INTERIMDATA = SPATIALIZED_FOLDER + 'CCI_spatialization_interimdata.gpkg'

    ## Parameter Check
    # Create folders to store the data results
    DIR_DATA = "data/"
    DIR_VAR = DIR_DATA + "processed/{}/".format("CCI")
    DIR_INDEX = DIR_VAR + "03_index/"

    if SAVE_FIGS is True or SAVE_INDEX is True:
        folder_list = [INDEX_FOLDER, DIR_VAR, DIR_INDEX]

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    ## Preprocessing
    # read spatialized dataset into geodataframe
    gdf = gpd.read_file(SPATIALIZED_INTERIMDATA)
    gdf.head()

    # Fill NAN-values with 0.
    gdf = gdf.fillna(0)

    ## Data Computation
    ### Benchmarks
    # Define benchmark per KPI - Italian CCI
    def bench(KPI):
        # Define benchmark per KPI
        if KPI == 'D1' or KPI == 'D2' or KPI == 'D3' or KPI == 'D4' or KPI == 'ECR1' or KPI == 'ECR2' or KPI == 'M2' or KPI == 'M4' or KPI == 'W3':
            return 1.0
        elif KPI == 'ECR3':
            return 0.55
        elif KPI == 'ECR4' or KPI == 'ECR5':
            return 40.0
        elif KPI == 'ECR6' or KPI == 'W1':
            return 0.0
        elif KPI == 'M1':
            return 900.0
        elif KPI == 'M3':
            return 100.0
        elif KPI == 'W2':
            return 0.0 # original 65.0
        else:
            pass

    ### Normalization - Area Values Computation
    def normalization(KPI):
        standard_column_KPI = KPI + '_standardized'
        NON_STANDARD_VALUE = gdf[KPI]

        # 1. Binary values 
        if KPI == 'D1' or KPI == 'D2' or KPI == 'D4' or KPI == 'ECR1' or KPI == 'ECR2'  or KPI == 'W3':
            gdf[standard_column_KPI] = NON_STANDARD_VALUE * 1.0

        # 2. Percentage values 
        elif KPI == 'D3' or KPI == 'ECR3' or KPI == 'ECR6' or KPI == "W2":
            if bench(KPI) > 0.0:
                KPI_standard_percentage = NON_STANDARD_VALUE * bench(KPI)
            elif bench(KPI) == 0.0:
                KPI_standard_percentage = (NON_STANDARD_VALUE - NON_STANDARD_VALUE.min()) / (NON_STANDARD_VALUE.max() - NON_STANDARD_VALUE.min())

            gdf[standard_column_KPI] = KPI_standard_percentage
            
        # 3. Threshold_down values
        elif KPI == 'ECR4' or KPI == 'ECR5':
            column = []

            for i in NON_STANDARD_VALUE:
                if (i >= -1.0) & (i < 0.5 * bench(KPI)):
                    column.append(4)
                elif (i >= 0.5 * bench(KPI)) & (i < bench(KPI)): 
                    column.append(3)
                elif (i >= bench(KPI)) & (i < 1.5 * bench(KPI)):
                    column.append(2)
                elif (i >= 1.5 * bench(KPI)) & (i < 2.0 * bench(KPI)): 
                    column.append(1)
                elif (i >= 2.0 * bench(KPI)):
                    column.append(0)
                else: 
                    column.append(0)
            gdf[standard_column_KPI] = column
            
            # Normalize given threshold values
            gdf[standard_column_KPI] = (gdf[standard_column_KPI] - gdf[standard_column_KPI].min()) / (gdf[standard_column_KPI].max() - gdf[standard_column_KPI].min())  

        # 4. Threshold_up values
        elif KPI == 'M1' or KPI == 'M2' or KPI == 'M3' or KPI == 'M4':
            KPI_standard_thresholdup = NON_STANDARD_VALUE / bench(KPI)
            gdf[standard_column_KPI] = KPI_standard_thresholdup
            gdf[standard_column_KPI].values[gdf[standard_column_KPI].values > 1.0] = 1.0
        
        # 5. Quartile_down values
        elif KPI == 'W1':
            column = []
            quartiles = np.quantile(NON_STANDARD_VALUE, [0.25,0.5,0.75])
            
            for i,j in enumerate(NON_STANDARD_VALUE):
                if NON_STANDARD_VALUE < quartiles[0]:
                    column.append(4)
                elif quartiles[0] <= NON_STANDARD_VALUE < quartiles[1]:
                    column.append(3)
                elif quartiles[1] <= NON_STANDARD_VALUE < quartiles[2]:
                    column.append(1)
                elif quartiles[2] <= NON_STANDARD_VALUE:
                    column.append(0)
                else:
                    column.append(0)

            gdf[standard_column_KPI] = column

            # Normalize given threshold values
            gdf[standard_column_KPI] = (gdf[standard_column_KPI] - gdf[standard_column_KPI].min()) / (gdf[standard_column_KPI].max() - gdf[standard_column_KPI].min())  

        return gdf[standard_column_KPI]

    def alternative_normalization(KPI):
        standard_column_KPI = KPI + '_standardized'
        NON_STANDARD_VALUE = gdf[KPI]

        # 1. Binary values 
        if KPI == 'D1' or KPI == 'D2' or KPI == 'D4' or KPI == 'ECR1' or KPI == 'ECR2'  or KPI == 'W3':
            gdf[standard_column_KPI] = NON_STANDARD_VALUE * 1.0

        # 2. Percentage values 
        elif KPI == 'D3' or KPI == 'ECR3' or KPI == 'ECR6':
            if bench(KPI) > 0.0:
                KPI_standard_percentage = NON_STANDARD_VALUE * bench(KPI)
            elif bench(KPI) == 0.0:
                KPI_standard_percentage = (NON_STANDARD_VALUE - NON_STANDARD_VALUE.min()) / (NON_STANDARD_VALUE.max() - NON_STANDARD_VALUE.min())

            gdf[standard_column_KPI] = KPI_standard_percentage
            
        # 3. Threshold_down values
        elif KPI == 'ECR4' or KPI == 'ECR5':
            column = []

            for i in NON_STANDARD_VALUE:
                if (i >= -1.0) & (i < 0.5 * bench(KPI)):
                    column.append(4)
                elif (i >= 0.5 * bench(KPI)) & (i < bench(KPI)): 
                    column.append(3)
                elif (i >= bench(KPI)) & (i < 1.5 * bench(KPI)):
                    column.append(2)
                elif (i >= 1.5 * bench(KPI)) & (i < 2.0 * bench(KPI)): 
                    column.append(1)
                elif (i >= 2.0 * bench(KPI)):
                    column.append(0)
                else: 
                    column.append(0)
            gdf[standard_column_KPI] = column
            
            # Normalize given threshold values
            gdf[standard_column_KPI] = (gdf[standard_column_KPI] - gdf[standard_column_KPI].min()) / (gdf[standard_column_KPI].max() - gdf[standard_column_KPI].min())  

        # 4. Threshold_up values
        elif KPI == 'M1' or KPI == 'M2' or KPI == 'M3' or KPI == 'M4':
            KPI_standard_thresholdup = NON_STANDARD_VALUE / bench(KPI)
            gdf[standard_column_KPI] = KPI_standard_thresholdup
            gdf[standard_column_KPI].values[gdf[standard_column_KPI].values > 1.0] = 1.0

        # 5. Quartile_down values
        elif KPI == 'W1':
            column = []
            quartiles = np.quantile(NON_STANDARD_VALUE, [0.25,0.5,0.75])
            
            for i,j in enumerate(NON_STANDARD_VALUE):
                if j < quartiles[0]:
                    column.append(4)
                elif quartiles[0] <= j < quartiles[1]:
                    column.append(3)
                elif quartiles[1] <= j < quartiles[2]:
                    column.append(1)
                elif quartiles[2] <= j:
                    column.append(0)
                else:
                    column.append(0)

            gdf[standard_column_KPI] = column

            # Normalize given threshold values
            gdf[standard_column_KPI] = (gdf[standard_column_KPI] - gdf[standard_column_KPI].min()) / (gdf[standard_column_KPI].max() - gdf[standard_column_KPI].min())  

        # 6. Quartile_up values
        elif KPI == 'W2':
            column = []
            quartiles = np.quantile(NON_STANDARD_VALUE, [0.25,0.5,0.75])
            
            for i,j in enumerate(NON_STANDARD_VALUE):
                if j < quartiles[0]:
                    column.append(0)
                elif quartiles[0] <= j < quartiles[1]:
                    column.append(1)
                elif quartiles[1] <= j < quartiles[2]:
                    column.append(3)
                elif quartiles[2] <= j:
                    column.append(4)
                else:
                    column.append(0)

            gdf[standard_column_KPI] = column

            # Normalize given threshold values
            gdf[standard_column_KPI] = (gdf[standard_column_KPI] - gdf[standard_column_KPI].min()) / (gdf[standard_column_KPI].max() - gdf[standard_column_KPI].min())  


        return gdf[standard_column_KPI]

    # Calculate benchmark and normalized values per KPI
    for KPI in KPI_list:    
        if ALTERNATIVE_AREA_COMPUTATION is True:
            bench(KPI)
            alternative_normalization(KPI)
        
        else:
            bench(KPI)
            normalization(KPI)

    gdf

    ## Data Weighting
    ### KPI Weighting
    # Create function to define weight per KPI
    def KPI_weight(KPI):
        if KPI == 'W1' or KPI == 'W2':
            KPI_weight = 0.4
        elif KPI == 'D1' or KPI == 'D2'or KPI == 'D3' or KPI == 'ECR3' or KPI == 'M2' or KPI == 'M4':
            KPI_weight = 0.3
        elif KPI == 'ECR1' or KPI == 'ECR2' or KPI == 'M1' or KPI == 'M3' or KPI == 'W3':
            KPI_weight = 0.2  
        elif KPI == 'D4' or KPI == 'ECR4' or KPI == 'ECR5' or KPI == 'ECR6':
            KPI_weight = 0.1    
        else:
            pass

        return KPI_weight

    # Create alternative function to define weight per KPI, when not all indicators are incorporated
    ## Calculated to original percentage
    def alternative_KPI_weight(KPI):
        if KPI == 'ECR1' or KPI == 'ECR2' or KPI == 'W3':
            KPI_weight = 2/6 # 0.3333333
        elif KPI == 'ECR4' or KPI == 'ECR5':
            KPI_weight = 1/6 # 0.1666666 
        elif KPI == 'W2':
            KPI_weight = 4/6 # 0.6666666
        elif KPI == 'M1' or KPI == 'M3':
            KPI_weight = 0.2   
        elif KPI == 'D1' or KPI == 'D2'or KPI == 'D3' or KPI == 'M2' or KPI == 'M4':
            KPI_weight = 0.3
        elif KPI == 'D4':
            KPI_weight = 0.1
        else:
            pass

        return KPI_weight

    # Calculate weight per KPI
    for KPI in KPI_list:
        standard_column_KPI = KPI + '_standardized'
        weighted_KPI = KPI + '_w'

        gdf[weighted_KPI]= gdf[standard_column_KPI] * KPI_weight(KPI)

        if ALTERNATIVE_KPI_WEIGHT == True:
            gdf[weighted_KPI]= gdf[standard_column_KPI] * alternative_KPI_weight(KPI)

    gdf

    ### Level Weighting
    # Create columns for level weights
    gdf["Digitalization"] = 0
    gdf["Energy_Climate_Resources"] = 0
    gdf["Mobility"] = 0
    gdf["Waste"] = 0

    # Define weights of CCI levels
    for KPI in KPI_list:
        
        weighted_KPI = KPI + '_w'
        leveled_KPI = 'temporal_' + weighted_KPI

        # Define weighting per KPI level
        if KPI[0] == 'D':
            gdf["Digitalization"] = gdf["Digitalization"] + gdf[weighted_KPI]
        elif KPI[0] == 'E':
            gdf["Energy_Climate_Resources"] = gdf["Energy_Climate_Resources"] + gdf[weighted_KPI]
        elif KPI[0] == 'M':
            gdf["Mobility"] = gdf["Mobility"] + gdf[weighted_KPI]
        elif KPI[0] == 'W':
            gdf["Waste"] = gdf["Waste"] + gdf[weighted_KPI]
        else:
            pass

    gdf

    ## Index Implementation
    # Calculate total weighted value per municipality
    gdf["CCI"] = gdf["Digitalization"] * 0.2 + gdf["Energy_Climate_Resources"] * 0.3 + gdf["Mobility"] * 0.2 + gdf["Waste"] * 0.3

    gdf

    # filter columns that refer to weighted values
    gdf_master = gdf[[
    'CTOT',
    'CMUN',
    'Municipality',
    'CCI',
    'Digitalization',
    'Energy_Climate_Resources',
    'Mobility',
    'Waste',
    'D1_w',
    'D2_w',
    'D3_w',
    'D4_w',
    'ECR1_w',
    'ECR2_w',
    'ECR4_w',
    'ECR5_w',
    'M1_w',
    'M2_w',
    'M3_w',
    'M4_w',
    'W2_w', 
    'W3_w', 
    'geometry'
    ]]

    gdf_master

    # Drop duplicate values (contain same values)
    gdf_master = gdf_master.drop_duplicates(subset=['CTOT'])
    gdf_master

    ### Total Index
    # Plot index results - Circular City Index (Total)
    fig, ax = plt.subplots(figsize=(20, 20))

    gdf_master.plot(
        ax=ax,
        column="CCI",
        #edgecolor="black",
        legend=True,
        figsize=(20, 20),
        cmap="RdYlGn",
        legend_kwds={"shrink": 0.7},
    )

    ax.set_title("Circular City Index", fontsize=20, y=1.01)

    if SAVE_FIGS is True:
        plt.savefig(DIR_INDEX + "index_results_CCI_total.svg", format="svg")

    plt.show()

    ## Export files
    # exports the geodataframe into GeoPackage file
    file_name = 'CCI_Index'
    data_format = '.gpkg'

    export_name = file_name + data_format

    if SAVE_INDEX is True:
        gdf_master.to_file(INDEX_FOLDER + export_name, driver='GPKG') 

    # exports the geodataframe into Shapefile
    file_name = 'CCI_Index'
    data_format = '.shp'

    export_name = file_name + data_format

    if SAVE_INDEX is True:
        gdf_master.to_file(DIR_INDEX + export_name)

    # exports the dataframe into csv file
    file_name = 'CCI_Index'
    data_format = '.csv'

    export_name = file_name + data_format

    if SAVE_INDEX is True:
        gdf_master.to_csv(DIR_INDEX + export_name)

    print("Circular City Index is computed")
    print("##################################")
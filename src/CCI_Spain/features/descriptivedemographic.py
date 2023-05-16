import seaborn as sns
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def DescriptiveDemographic(VARIABLE_TO_PREDICT, AREA_TO_PREDICT):
    # papermill parameters cell
    OUTPUT_WARNINGS = False
    SAVE_FIGS = False
    SAVE_TABLES = False

    if OUTPUT_WARNINGS is False:
        import warnings

        warnings.filterwarnings("ignore")

    AREA_TO_PREDICT_dict = {
        "Andalusia": "01", 
        "Aragon": "02",
        "Asturias": "03", 
        "Balearic Islands": "04",
        "Canarias": "05", 
        "Cantabria": "06",
        "Castile and Leon": "07", 
        "Castille-La Mancha": "08",
        "Catalonia": "09", 
        "Valencia": "10",
        "Extremadura": "11", 
        "Galicia": "12",
        "Madrid": "13", 
        "Murcia": "14",
        "Navarre": "15", 
        "Basque Country": "16",
        "La Rioja": "17",
        "Ceuta": "18",
        "Melilla": "19",
        "Minor Plazas de Soberanía": "20",
        }

    ### Data
    # Datasets 
    INDEX_DATA = 'data/processed/CCI/03_index/CCI_Index.gpkg'
    DEMOGRPAHIC_DATA = '/work/data/interim/demographic_interimdata/merged_demographic_interimdata/Spatial_demographic_interimdata.gpkg'
    ## Parameter check

    # Create folders to store the data
    DIR_DATA = "data/"
    DIR_VAR = DIR_DATA + "processed/{}/".format("CCI")
    DIR_DEMOGRAPHIC = DIR_VAR + "11_demographic/{}/".format(AREA_TO_PREDICT)

    if SAVE_FIGS:
        folder_list = [
            DIR_DEMOGRAPHIC,
            DIR_DEMOGRAPHIC + "coefficients",
        ]

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    PATH_TO_FILE = DIR_DATA + "interim/demographic_interimdata/merged_demographic_interimdata/Spatial_demographic_interimdata.gpkg"
    if os.path.isfile(PATH_TO_FILE) is False:
        raise Exception(
                    'Please run first the notebooks with the same area and "SAVE_DATA" set to True: /n"00acquisition.ipynb", /n"01datamanagement.ipynb", /n"02dataspatialization.ipynb", /n"03index.ipynb"'
        )

    ## Plot results of Index
    # Read spatial dataset into GeoDataFrame
    gdf = gpd.read_file(DEMOGRPAHIC_DATA) 

    # Reset index to column
    gdf.reset_index()

    # Choose Study Area
    if AREA_TO_PREDICT in AREA_TO_PREDICT_dict:
        gdf = gdf[gdf["CTOT"].astype(str).str.contains(r'^' + AREA_TO_PREDICT_dict[AREA_TO_PREDICT])]
    elif AREA_TO_PREDICT == "Iberian Pensinula":
        gdf = gdf[~gdf.CTOT.str.contains(r'^04')] # --> DROP BALEARIC ISLANDS
        gdf = gdf[~gdf.CTOT.str.contains(r'^05')] # --> DROP CANARIAS
        gdf = gdf[~gdf.CTOT.str.contains(r'^18')] # --> DROP CEUTA
        gdf = gdf[~gdf.CTOT.str.contains(r'^19')] # --> DROP MELILLA
        gdf = gdf[~gdf.CTOT.str.contains(r'^20')] # --> DROP MINOR PLAZAS DE SOBERINIA
    elif AREA_TO_PREDICT == "Spain":
        pass

    # Redo index by CTOT
    gdf.set_index("CTOT", inplace=True)

    # Read spatial dataset into GeoDataFrame
    CCI = gpd.read_file(INDEX_DATA) 

    # Reset index to column
    CCI.reset_index()

    # Choose Study Area
    if AREA_TO_PREDICT in AREA_TO_PREDICT_dict:
        CCI = CCI[CCI["CTOT"].astype(str).str.contains(r'^' + AREA_TO_PREDICT_dict[AREA_TO_PREDICT])]
    elif AREA_TO_PREDICT == "Iberian Pensinula":
        CCI = CCI[~CCI.CTOT.str.contains(r'^04')] # --> DROP BALEARIC ISLANDS
        CCI = CCI[~CCI.CTOT.str.contains(r'^05')] # --> DROP CANARIAS
        CCI = CCI[~CCI.CTOT.str.contains(r'^18')] # --> DROP CEUTA
        CCI = CCI[~CCI.CTOT.str.contains(r'^19')] # --> DROP MELILLA
        CCI = CCI[~CCI.CTOT.str.contains(r'^20')] # --> DROP MINOR PLAZAS DE SOBERINIA
    elif AREA_TO_PREDICT == "Spain":
        pass

    # Redo index by CTOT
    CCI.set_index("CTOT", inplace=True)

    CCI = pd.merge(gdf, CCI[VARIABLE_TO_PREDICT], left_index=True, right_index=True)
    CCI = CCI[[VARIABLE_TO_PREDICT, "geometry"]]

    DEMOGRAPHIC_list = [
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
    ]

    SOCIOECONOMIC_list = [
        "HOUSING_AVERAGE_HOUSEHOLD_SIZE_2020",
        "HOUSING_PERC_SINGLEPERSON_HOUSEHOLD_2020",
        'HOUSING_RESIDENT_BUILDINGS_PER_CAPITA_2011',
        "INCOME_PER_CAPITA_2020",
        "INCOME_PER_HOUSEHOLD_2020",
        'INCOME_PERC_UNEMPLOYMENT_BENEFITS_OF_AVERAGE_SALARY_2020',
    ]
    
    ECONOMIC_list = [
        "WEALTH_GINI_2020",
        "DEBT_MUNICIPALITY_PER_CAPITA_2021",
        "ECONOMY_COMPANIES_PER_CAPITA_2020",
        'AGRI_LIVESTOCKUNITS_DENSITY_KM2_2020',
        'AGRI_CATTLEFARMS_DENSITY_KM2_2020',    
        'TOURISM_HOUSES_PER_CAPITA_2022',
    ]
    ## Descriptive Statistics
    ### All Features
    # Description of all feature data
    if SAVE_TABLES is True:
        gdf.describe().to_csv((DIR_DEMOGRAPHIC + "descriptive_all_features.csv"), index=True)

    gdf.describe()

    # Preprocess data
    X = gdf.drop(columns=['geometry'])
    X['CCI'] = CCI['CCI']
    imputer = KNNImputer()
    X_transformed = imputer.fit_transform(X)
    X = pd.DataFrame(X_transformed, columns=X.columns)

    # Calculate Bivariate Regression Correlations of CCI with each feature
    correlation_matrix = X.corr()

    # Calculate p-values
    p_values = []
    for col in correlation_matrix.columns:
        _, p = stats.pearsonr(X[col], X['CCI'])
        p_values.append(p)

    # Add p-values to the correlation matrix
    correlation_matrix.loc['p-value'] = p_values
    print(correlation_matrix)

    if SAVE_TABLES is True:
        correlation_matrix['CCI'].to_csv((DIR_DEMOGRAPHIC + "bivariate_correlations_CCI_all_features.csv"), index=True)

    correlation_matrix

    # Boxplot of all feature data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    X_normalized = pd.DataFrame(data=X_normalized, columns=X.columns)

    sns.set_style('whitegrid')
    sns.boxplot(
        x="value", 
        y="variable",
        showfliers = False, 
        data=pd.melt(X_normalized)
        ).set(
            title='Boxplot of Socio-Demographic and Economic Features',
            xlabel='Variable', 
            ylabel='Value'
            )

    # Add vertical line
    plt.axvline(x=0.5, color='r')

    if SAVE_FIGS is True:
        plt.savefig(DIR_DEMOGRAPHIC + "boxplot_all_features.svg", format="svg")

    plt.show()

    # Distribution of all feature data
    fig = sns.displot(
        data=X_normalized, 
        kind="kde",
        fill=True).set(
            title='Distribution of Socio-Demographic and Economic Features', 
            xlabel='Value', 
            ylabel='Density'
            )

    if SAVE_FIGS is True:
        plt.savefig(DIR_DEMOGRAPHIC + "distribution_all_features.svg", format="svg")
    ### Best Model Features

    # Load best model dataset from 07_linear notebook
    linear_coefs = pd.read_csv("data/processed/CCI/Spain/07_linear/coefficients.csv", index_col=0)
    best_model = linear_coefs.drop(["Intercept"], axis=1).columns

    # Use best model in geodataframe
    gdf_best_model = gdf[best_model]
    gdf_best_model['geometry'] = gdf['geometry']

    print(best_model)

    # Description of all feature data
    if SAVE_TABLES is True:
        gdf_best_model.describe().to_csv((DIR_DEMOGRAPHIC + "descriptive_selected_features.csv"), index=True)

    gdf_best_model.describe()

    # Boxplot of all feature data
    sns.set_style('whitegrid')
    sns.boxplot(
        x="value", 
        y="variable", 
        data=pd.melt(gdf_best_model)
        ).set(
            title='Boxplot of Socio-Demographic and Economic Features',
            xlabel='Variable', 
            ylabel='Value'
            )

    # Add vertical line
    plt.axvline(x=0.5, color='r')

    if SAVE_FIGS is True:
        plt.savefig(DIR_DEMOGRAPHIC + "boxplot_selected_features.svg", format="svg")

    plt.show()

    # Distirbution of all feature data
    fig = sns.displot(
        data=gdf_best_model, 
        kind="kde", 
        fill=True).set(
            title='Distribution of Socio-Demographic and Economic Features', 
            xlabel='Value', 
            ylabel='Density'
            )

    if SAVE_FIGS is True:
        plt.savefig(DIR_DEMOGRAPHIC + "distribution_selected_features.svg", format="svg")

    ## Mapping (Visualization)
    # Define line colors
    def line_color(area):
        if area == "Spain" or area == "Iberian Pensinula": color = 'face'
        else: color = "black"
        return color

    # Define scheme type
    def scheme_type(variable):
        if variable.startswith("PERC_"): scheme =  'naturalbreaks'
        else: scheme = "naturalbreaks"
        return scheme

    # Define map color type
    def map_color(variable):
        if variable.startswith("PERC_"): color =  'coolwarm'
        else: color = "Blues" #YlOrBr
        return color

    ### All Features
    cluster_variables = list(gdf)[:-1]
    nrows_clusters = math.ceil(len(cluster_variables) / 3)

    f, axs = plt.subplots(nrows=nrows_clusters, ncols=3, figsize=(12, 12))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Start a loop over all the variables of interest
    for i, col in enumerate(cluster_variables):
        # select the axis where the map will go
        ax = axs[i]
        # Plot the map
        gdf.plot(
            column=col,
            ax=ax,
            edgecolor=line_color(AREA_TO_PREDICT),
            scheme=scheme_type(col),
            linewidth=0,
            cmap=map_color(col),
        )
        # Remove axis clutter
        ax.set_axis_off()
        # Set the axis title to the name of variable being plotted
        ax.set_title(col, fontsize=8)

    if SAVE_FIGS:
        plt.savefig(DIR_DEMOGRAPHIC + "map_all_features.svg", format="svg")

    plt.tight_layout()

    # Display the figure
    plt.show()

    ### Beste Model Features
    cluster_variables = list(gdf_best_model)[:-1]
    nrows_clusters = math.ceil(len(cluster_variables) / 3)

    f, axs = plt.subplots(nrows=nrows_clusters, ncols=3, figsize=(12, 12))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Start a loop over all the variables of interest
    for i, col in enumerate(cluster_variables):
        # select the axis where the map will go
        ax = axs[i]
        # Plot the map
        gdf_best_model.plot(
            column=col,
            ax=ax,
            edgecolor=line_color(AREA_TO_PREDICT),
            scheme=scheme_type(col),
            linewidth=0,
            cmap=map_color(col),
        )
        # Remove axis clutter
        ax.set_axis_off()
        # Set the axis title to the name of variable being plotted
        ax.set_title(col, fontsize=8)

    if SAVE_FIGS:
        plt.savefig(DIR_DEMOGRAPHIC + "map_selected_features.svg", format="svg")

    plt.tight_layout()

    # Display the figure
    plt.show()

    ### Total Index
    # Plot results 
    for col in gdf.columns:
        if col in DEMOGRAPHIC_list:
            fig, ax = plt.subplots(figsize=(20, 20))

            gdf.plot(
                ax=ax,
                column=col,
                edgecolor=line_color(AREA_TO_PREDICT),
                legend=True,
                figsize=(20, 20),
                #legend_kwds={"shrink": 0.7},
                scheme=scheme_type(col),            
                cmap=map_color(col)
            )

            ax.set_title("Demographic feature: {} - {}".format(col, AREA_TO_PREDICT), fontsize=20, y=1.01)

            if SAVE_FIGS is True:
                plt.savefig(DIR_DEMOGRAPHIC + "{}_{}".format(col, AREA_TO_PREDICT) + ".svg", format="svg")

            plt.show()
        else: 
            pass

    # Plot results 
    for col in gdf.columns:
        if col in SOCIOECONOMIC_list:
            fig, ax = plt.subplots(figsize=(20, 20))

            gdf.plot(
                ax=ax,
                column=col,
                edgecolor=line_color(AREA_TO_PREDICT),
                legend=True,
                figsize=(20, 20),
                cmap=map_color(col),
                #legend_kwds={"shrink": 0.7},
                scheme=scheme_type(col),
            )

            ax.set_title("Socio-Economic feature: {} - {}".format(col, AREA_TO_PREDICT), fontsize=20, y=1.01)

            if SAVE_FIGS is True:
                plt.savefig(DIR_DEMOGRAPHIC + "{}_{}".format(col, AREA_TO_PREDICT) + ".svg", format="svg")

            plt.show()
        else: 
            pass

    # Plot results 
    for col in gdf.columns:
        if col in ECONOMIC_list:
            fig, ax = plt.subplots(figsize=(20, 20))

            gdf.plot(
                ax=ax,
                column=col,
                edgecolor=line_color(AREA_TO_PREDICT),
                legend=True,
                figsize=(20, 20),
                cmap=map_color(col),
                #legend_kwds={"shrink": 0.7},
                scheme=scheme_type(col),
            )

            ax.set_title("Economic feature: {} - {}".format(col, AREA_TO_PREDICT), fontsize=20, y=1.01)

            if SAVE_FIGS is True:
                plt.savefig(DIR_DEMOGRAPHIC + "{}_{}".format(col, AREA_TO_PREDICT) + ".svg", format="svg")

            plt.show()
        else: 
            pass

    print("The Descriptive Analysis of the Demographic Data is computed for {}".format(AREA_TO_PREDICT))
    print("##################################")

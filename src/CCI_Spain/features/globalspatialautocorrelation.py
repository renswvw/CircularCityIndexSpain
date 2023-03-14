# 05. Data Analytics - Global Spatial Autocorrelation
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
import seaborn 
from splot.esda import plot_moran

def GlobalSpatialAutocorrelation(VARIABLE_TO_PREDICT, VARIABLE_TO_DROP, AREA_TO_PREDICT):
    # papermill parameters cell
    OUTPUT_WARNINGS = False
    SAVE_FIGS = True
    DROP_VARIABLE = True

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

    ## Parameter Check
    # Create folders to store the data
    DIR_DATA = "data/"
    DIR_VAR = DIR_DATA + "processed/{}/{}/".format(VARIABLE_TO_PREDICT, AREA_TO_PREDICT)
    DIR_GSA = DIR_VAR + "05_gsa/"

    if SAVE_FIGS:
        folder_list = [
            DIR_GSA,
            DIR_GSA + "coefficients",
        ]

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    PATH_TO_FILE = DIR_DATA + "interim/demographic_interimdata/merged_demographic_interimdata/Spatial_demographic_interimdata.csv"
    if os.path.isfile(PATH_TO_FILE) is False:
        raise Exception(
            'Please run first the notebooks with the same area and "SAVE_DATA" set to True: /n"00acquisition.ipynb", /n"01datamanagement.ipynb", /n"02dataspatialization.ipynb", /n"03index.ipynb"'
        )

    ## Target variable
    ### Dependent variable
    # Read CCI results
    CCI = pd.read_csv('data/processed/CCI/03_index/CCI_Index.csv')

    # Add extra digit to dataset['CTOT'] - if it contains less than 7 characters
    CCI['CTOT'] = CCI['CTOT'].apply(lambda x: '{0:0>7}'.format(x))

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

    CCI.set_index("CTOT", inplace=True)

    ### Independent variables
    # Read spatial dataset into GeoDataFrame
    geo_info = gpd.read_file(DIR_DATA + "interim/Spatialization_interimdata/CCI_spatialization_interimdata.gpkg")

    # Choose Study Area
    if AREA_TO_PREDICT in AREA_TO_PREDICT_dict:
        geo_info = geo_info[geo_info["CTOT"].astype(str).str.contains(r'^' + AREA_TO_PREDICT_dict[AREA_TO_PREDICT])]
    elif AREA_TO_PREDICT == "Iberian Pensinula":
        geo_info = geo_info[~geo_info.CTOT.str.contains(r'^04')] # --> DROP BALEARIC ISLANDS
        geo_info = geo_info[~geo_info.CTOT.str.contains(r'^05')] # --> DROP CANARIAS
        geo_info = geo_info[~geo_info.CTOT.str.contains(r'^18')] # --> DROP CEUTA
        geo_info = geo_info[~geo_info.CTOT.str.contains(r'^19')] # --> DROP MELILLA
        geo_info = geo_info[~geo_info.CTOT.str.contains(r'^20')] # --> DROP MINOR PLAZAS DE SOBERINIA
    elif AREA_TO_PREDICT == "Spain":
        pass

    geo_info.set_index("CTOT", inplace=True)
    geo_info = geo_info.drop(["Municipality"], axis=1,)

    if DROP_VARIABLE is True:
        geo_info = geo_info.drop(VARIABLE_TO_DROP, axis=1,)

    gdf = geo_info.drop(["CMUN", "POP21", "geometry"], axis=1,)

    # Add dependent variable column to spatial dataset
    geo_info[VARIABLE_TO_PREDICT] = CCI[VARIABLE_TO_PREDICT]

    # Add dependent variable column to spatial dataset
    id_max_KPI = geo_info[VARIABLE_TO_PREDICT].idxmax()

    # Define area with maximum value of dependent variable
    print("Area with maximum value: " + str(id_max_KPI))

    ### Plot target variable

    # Define line colors
    def line_color(area):
        if area == "Spain" or area == "Iberian Pensinula": color = 'face'
        else: color = "black"
        return color

    # Plot dependent variable (target variable)
    fig, ax = plt.subplots(figsize=(20, 20))

    geo_info.plot(
        ax=ax,
        column=VARIABLE_TO_PREDICT,
        edgecolor=line_color(AREA_TO_PREDICT),
        legend=True,
        figsize=(20, 20),
        cmap="RdYlGn",
        legend_kwds={"shrink": 0.7},
    )

    ax.set_title("Target variable: " + str(VARIABLE_TO_PREDICT), fontsize=20, y=1.01)

    if SAVE_FIGS:
        plt.savefig(DIR_GSA + "target_variable.svg", format="svg")

    plt.show()

    ## Spatial Weights Matrix
    # Generate W from the GeoDataFrame
    w = weights.distance.KNN.from_dataframe(geo_info, k=5)

    # Row-standardization
    w.transform = "R"

    ## Global Spatial Autocorrelation

    ### Spatial Lag
    # Apply geo_info spatial lag
    VARIABLE_TO_PREDICT_lag = VARIABLE_TO_PREDICT + '_lag'
    geo_info[VARIABLE_TO_PREDICT_lag] = weights.spatial_lag.lag_spatial(
        w, geo_info[VARIABLE_TO_PREDICT]
    )

    # Plot VARIABLE_TO_PREDICT normal vs VARIABLE_TO_PREDICT with spatial log
    f, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axs

    geo_info.plot(
        column=VARIABLE_TO_PREDICT,
        cmap="viridis",
        scheme="quantiles",
        k=5,
        edgecolor="white",
        linewidth=0.0,
        alpha=0.75,
        legend=True,    
        legend_kwds={'loc':'lower right'},
        ax=ax1,
    )
    ax1.set_axis_off()
    ax1.set_title(str(VARIABLE_TO_PREDICT))

    geo_info.plot(
        column=VARIABLE_TO_PREDICT_lag,
        cmap="viridis",
        scheme="quantiles",
        k=5,
        edgecolor="white",
        linewidth=0.0,
        alpha=0.75,
        legend=True,
        legend_kwds={'loc':'lower right'},
        ax=ax2,
    )
    ax2.set_axis_off()
    ax2.set_title(str(VARIABLE_TO_PREDICT) + " - Spatial Lag")

    if SAVE_FIGS:
        plt.savefig(DIR_GSA + "normalVSspatiallog.svg", format="svg")

    plt.show()

    ## Moran Plot and Moran's I
    VARIABLE_TO_PREDICT_std = VARIABLE_TO_PREDICT + "_std"
    VARIABLE_TO_PREDICT_lag_std = VARIABLE_TO_PREDICT + "_lag_std"

    # Calculate standard deviation vs standard deviation of log
    geo_info[VARIABLE_TO_PREDICT_std] = geo_info[VARIABLE_TO_PREDICT] - geo_info[VARIABLE_TO_PREDICT].mean()
    geo_info[VARIABLE_TO_PREDICT_lag_std] = (geo_info[VARIABLE_TO_PREDICT_lag] - geo_info[VARIABLE_TO_PREDICT_lag].mean())

    # Create Moran Plot
    f, ax = plt.subplots(1, figsize=(6, 6))
    seaborn.regplot(
        x=VARIABLE_TO_PREDICT_std,
        y=VARIABLE_TO_PREDICT_lag_std,
        ci=None,
        data=geo_info,
        line_kws={"color": "r"},
    )
    ax.axvline(0, c="k", alpha=0.5)
    ax.axhline(0, c="k", alpha=0.5)
    ax.set_title("Moran Plot - " + str(VARIABLE_TO_PREDICT))

    if SAVE_FIGS:
        plt.savefig(DIR_GSA + "MoranPlot.svg", format="svg")

    plt.show()

    w.transform = "R"
    moran = esda.moran.Moran(geo_info[VARIABLE_TO_PREDICT], w)

    # Calculate Moran's I
    moran.I
    print("Moran's I value = " + str(moran.I ))

    # Calculate Moran's I
    moran.p_sim
    print("Moran's I p-value significance = " + str(moran.p_sim))

    # Significance level
    if moran.p_sim <= 0.05:
        print("Moran's I is significant")
    elif moran.p_sim > 0.05:
        print("Moran's I is NOT significant")

    # Plot Moran Scatterplot
    plot_moran(moran, zstandard=False);

    if SAVE_FIGS:
        plt.savefig(DIR_GSA + "Moran_Scatterplot.svg", format="svg")

    print("Global Spatial Autocorrelation of {} is computed for {}".format(VARIABLE_TO_PREDICT, AREA_TO_PREDICT))
    print("##################################")
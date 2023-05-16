import seaborn as sns
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np; np.random.seed(42)

def DescriptiveAnalysis(AREA_TO_PREDICT):
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
    POP21_DATA = 'data/interim/POP21_interimdata.csv'

    ## Parameter check
    # Create folders to store the data
    DIR_DATA = "data/"
    DIR_VAR = DIR_DATA + "processed/{}/{}/".format("CCI", AREA_TO_PREDICT)
    DIR_RESULTS = DIR_VAR + "04_descriptiveanalysis/"

    if SAVE_FIGS or SAVE_TABLES:
        folder_list = [
            DIR_RESULTS,
        ]

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    PATH_TO_FILE = DIR_DATA + "processed/CCI/03_index/CCI_Index.csv"
    if os.path.isfile(PATH_TO_FILE) is False:
        raise Exception(
                    'Please run first the notebooks with the same area and "SAVE_DATA" set to True: /n"00acquisition.ipynb", /n"01datamanagement.ipynb", /n"02dataspatialization.ipynb", /n"03index.ipynb"'
        )

    # Read spatial dataset into GeoDataFrame
    gdf = gpd.read_file(INDEX_DATA) 

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

    ## Descriptive Statistics
    ### CCI & levels
    # Description of index data
    gdf_descriptive_main = gdf[['CCI','Digitalization', 'Energy_Climate_Resources', 'Mobility', 'Waste']]

    if SAVE_TABLES is True:
        gdf_descriptive_main.describe().to_csv((DIR_RESULTS + "descriptive_CCI_levels.csv"), index=True)

    gdf_descriptive_main.describe()
    gdf_descriptive_main.dtypes

    # Histogram of index data
    sns.set_style('whitegrid')
    fig = sns.histplot(
        data=gdf_descriptive_main, 
        x="CCI", 
        bins=100, 
        color='blue'
        ).set(
            title='Circular City Index', 
            xlabel='Index', 
            ylabel='n municipalities'
            )

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "histogram_CCI_Main.svg", format="svg")

    # Boxplot of index data
    df_boxplot = gdf_descriptive_main.rename(columns={"Digitalization":"D", "Energy_Climate_Resources":"ECR", 'Mobility':'M', 'Waste':'W'})

    sns.set_style('whitegrid')
    sns.boxplot(
        x="value", 
        y="variable", 
        data=pd.melt(df_boxplot)
        ).set(
            title='Boxplot of CCI Scores',
            xlabel='CCI Score', 
            ylabel='Variable'
            )

    # Add vertical line
    plt.axvline(x=0.5, color='r')

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "boxplot_CCI_Main.svg", format="svg")

    plt.show()

    # Distirbution of index data
    fig = sns.displot(
        data=df_boxplot, 
        kind="kde", 
        fill=True).set(
            title='Distribution of CCI Scores', 
            xlabel='CCI Score', 
            ylabel='Density'
            )

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "distribution_CCI_Main.svg", format="svg")

    ### KPIs
    # Description of index data (main columns)
    gdf_descriptive_KPIs = gdf.drop(columns=['CCI','Digitalization', 'Energy_Climate_Resources', 'Mobility', 'Waste','CMUN', 'Municipality', 'geometry'])

    if SAVE_TABLES is True:
        gdf_descriptive_KPIs.describe().to_csv((DIR_RESULTS + "descriptive_CCI_KPIs.csv"), index=True)

    gdf_descriptive_KPIs.describe()

    ### General Municipal Information
    # Load population dataframe
    df_population = pd.read_csv(POP21_DATA)

    # Add an extra 0 to CMUN
    df_population["CMUN"] = df_population["CMUN"].apply(lambda x: '{0:0>5}'.format(x))

    # Create categories per groups
    group = [0, 5000, 15000, 100000, df_population['POP21'].max()]

    # Create labels for population groups
    legenda = ['<5k', '5k-15k', '15k-100k', '>100k']

    # Classify population by group
    df_population['population_class'] = pd.cut(df_population['POP21'], bins=group, labels=legenda)

    # Classify by sum, total population and percentage of population
    population_class = df_population.groupby('population_class').agg({'POP21': ['count', 'sum']})
    population_class.columns = ['Number of Municipalities', 'Total Population']
    population_class['Percentage of Population'] = (population_class['Total Population'] / population_class['Total Population'].sum()) * 100

    if SAVE_TABLES is True:
        population_class.describe().to_csv(DIR_RESULTS + "descriptive_PopulationGroups.csv")

    population_class

    # Merge index data with population dataframe
    gdf_MUNsize = pd.merge(gdf, df_population)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot distribution by municipality class
    sns.kdeplot(
        data=gdf_MUNsize, 
        ax=ax,
        x="CCI", 
        hue="population_class", 
        fill=True, 
        common_norm=False, 
        alpha=0.4)

    if SAVE_FIGS:
        plt.savefig(DIR_RESULTS + "descriptive_density_levels.csv", format="svg")

    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.02), ncol=4, title=None, frameon=True)
        
    # Display the figure
    plt.show()

    levels_variables = ['Digitalization', 'Energy_Climate_Resources', 'Mobility', 'Waste']
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Start a loop over all the variables of interest
    for i, col in enumerate(levels_variables):
        # select the axis where the map will go
        ax = axs[i]
        print(i)
        # Plot distribution by municipality class
        sns.kdeplot(
            data=gdf_MUNsize, 
            ax=ax,
            x=col, 
            hue="population_class", 
            fill=True, 
            common_norm=False, 
            alpha=0.4)
        
        # Set the axis title to the name of variable being plotted
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.), ncol=4, title=None, frameon=True)
    
    if SAVE_FIGS:
        plt.savefig(DIR_RESULTS + "descriptive_density_levels.csv", format="svg")

    # Display the figure
    plt.show()

    # divide the dataframe into quartiles based on the CCI
    gdf_MUNsize['Quartile'] = pd.qcut(gdf_MUNsize['CCI'], 4, labels=False)

    # create a new dataframe to store the quartile data
    quartile_data = pd.DataFrame(columns=['Quartile', 'POP21', 'Quartile Range', 'Average POP21'])

    # loop through the quartiles and calculate the requested data
    for i in range(4):
        # select the subset of data for the current quartile
        quartile_df = gdf_MUNsize[gdf_MUNsize['Quartile'] == i]
        # calculate the POP21 for the quartile
        total_inhabitants = quartile_df['POP21'].sum()
        # calculate the quartile range
        quartile_range = f'{quartile_df["CCI"].min()} - {quartile_df["CCI"].max()}'
        # calculate the average POP21 for the quartile
        average_population_size = quartile_df['POP21'].mean()
        # add the quartile data to the quartile_data dataframe
        quartile_data.loc[i] = [f'Quartile {i+1}', total_inhabitants, quartile_range, average_population_size]

    # print the resulting quartile data
    quartile_data

    ## Province Capitals
    provincial_capitals = {
        'Álava': '01059',
        'Albacete': '02003',
        'Alicante': '03014',
        'Almería': '04013',
        'Asturias': '33044',
        'Ávila': '05019',
        'Badajoz': '06015',
        'Barcelona': '08019',
        'Burgos': '09059',
        'Cáceres': '10037',
        'Cádiz': '11012',
        'Cantabria': '39075',
        'Castellón': '12040',
        'Ciudad Real': '13038',
        'Córdoba': '14021',
        'Cuenca': '16032',
        'Girona': '17079',
        'Granada': '18087',
        'Guadalajara': '19130',
        'Guipúzcoa': '20069',
        'Huelva': '21041',
        'Huesca': '22135',
        'Islas Baleares': '07040',
        'Jaén': '23050',
        'La Coruña': '15030',
        'La Rioja': '26089',
        'Las Palmas': '35016',
        'León': '24089',
        'Lérida': '25120',
        'Lugo': '27028',
        'Madrid': '28079',
        'Málaga': '29067',
        'Murcia': '30030',
        'Navarra': '31201',
        'Orense': '32054',
        'Palencia': '34120',
        'Pontevedra': '36038',
        'Salamanca': '37274',
        'Santa Cruz de Tenerife': '38038',
        'Segovia': '40004',
        'Sevilla': '41091',
        'Soria': '42173',
        'Tarragona': '43148',
        'Teruel': '44190',
        'Toledo': '45168',
        'València': '46250',
        'Valladolid': '47186',
        'Vizcaya': '48020',
        'Zamora': '49275',
        'Zaragoza': '50297'
    }

    def capital_score(index_attribute):
        # Create a list of the province codes
        province_codes = list(set([municipality[:2] for municipality in gdf['CMUN']]))

        # Create an empty DataFrame to store the results
        results = pd.DataFrame(columns=['Province', 'Capital Score', 'Avg Other Municipalities CCI', 'Difference'])

        # Loop through each province code 
        for code in province_codes:
            # Get the province name from the provincial_capitals dictionary
            province = [key for key, value in provincial_capitals.items() if value.startswith(code)]
            if not province:
                continue
            province = province[0]
            # Get the CCI value of the provincial capital
            capital_cci = gdf.loc[gdf['CMUN'] == provincial_capitals[province], index_attribute].values[0]
            # Get the CCI values of the other municipalities in the province
            other_municipalities_cci = gdf.loc[gdf['CMUN'].str.startswith(code) & ~gdf['CMUN'].isin(provincial_capitals.values()), index_attribute]
            # Calculate the average CCI value of the other municipalities
            avg_other_municipalities_cci = other_municipalities_cci.mean()
            # Calculate the difference between the capital CCI and the average other municipalities CCI
            difference = capital_cci - avg_other_municipalities_cci
            # Create a new DataFrame with the results
            new_row = pd.DataFrame({'Province': [province], 'Capital Score': [capital_cci], 'Avg Other Municipalities CCI': [avg_other_municipalities_cci], 'Difference': [difference]})
            # Concatenate the new row with the results DataFrame
            results = pd.concat([results, new_row], ignore_index=True)

        # Sort the results by the difference in ascending order
        results = results.sort_values('Difference')

        # Create a scatter plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(x=results['Province'], y=results['Difference'], s=50, color='blue')

        # Add a red line at 0.0
        ax.axhline(y=0.0, color='red', linestyle='--')

        # Set axis labels and title
        ax.set_xlabel('Province')
        ax.set_ylabel('Difference')
        ax.set_title('Differences between Provincial Capital and Surrounding Municipalities - {}'.format(index_attribute))

        # Rotate x-axis labels
        plt.xticks(rotation=90)

        # Show the plot
        plt.show()

    # Run and Plot Differences between Capital and Rest of Provinces
    CCI_area = 'CCI'
    capital_score(CCI_area)

    ## Mapping (Visualization)

    # Define line colors
    def line_color(area):
        if area == "Spain" or area == "Iberian Pensinula": color = "face"
        else: color = "black"
        return color

    ### Total Index
    # Plot index results - Circular City Index (Total)
    fig, ax = plt.subplots(figsize=(20, 20))

    gdf.plot(
        ax=ax,
        column="CCI",
        edgecolor=line_color(AREA_TO_PREDICT),
        legend=True,
        legend_kwds={'loc':'lower right'},
        figsize=(20, 20),
        cmap="Blues", #cmap="RdYlGn",
        scheme="NaturalBreaks", #scheme="Quantiles",
        k=5 #k=4
    )

    # Set the axis title to the name of variable being plotted
    ax.set_title("Circular City Index \nScheme = Natural Breaks", fontsize=20, y=1.01)
    # Remove axis clutter
    ax.set_axis_off()

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "map_results_CCI.svg", format="svg")

    plt.show()
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 40))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    ## AXIS 0 ##
    # Get mean values per Province (CPRO)
    gdf_Prov = gdf.copy()
    gdf_Prov = gdf_Prov[["CCI","Digitalization", "Energy_Climate_Resources", "Mobility", "Waste", "geometry"]]
    gdf_Prov = gdf_Prov.dissolve(by=gdf_Prov.index.get_level_values('CTOT').str[0:4], aggfunc='mean')

    # select the axis where the map will go
    ax = axs[0]
    # Plot the map
    gdf_Prov.plot(
        ax=ax,
        column="CCI",
        edgecolor=None,
        legend=True,
        legend_kwds={'loc':'lower right'},
        figsize=(20, 20),
        cmap="Blues", #cmap="RdYlGn",
        scheme="NaturalBreaks", #scheme="Quantiles",
        k=5 #k=4
    )

    # Set the axis title to the name of variable being plotted
    ax.set_title("Circular City Index \nProvinces\nScheme = Quantiles", fontsize=20, y=1.01)
    # Remove axis clutter
    ax.set_axis_off()

    ## AXIS 1 ##
    # Get mean values per Autonomous Community (CAUC)
    gdf_AutComm = gdf.copy()
    gdf_AutComm = gdf_AutComm[["CCI","Digitalization", "Energy_Climate_Resources", "Mobility", "Waste", "geometry"]]
    gdf_AutComm = gdf_AutComm.dissolve(by=gdf_AutComm.index.get_level_values('CTOT').str[0:2], aggfunc='mean')

    # select the axis where the map will go
    ax = axs[1]
    # Plot the map
    gdf_AutComm.plot(
        ax=ax,
        column="CCI",
        edgecolor=None,
        legend=True,
        legend_kwds={'loc':'lower right'},
        figsize=(20, 20),
        cmap="Blues", #cmap="RdYlGn",
        scheme="NaturalBreaks", #scheme="Quantiles",
        k=5 #k=4
    )

    # Set the axis title to the name of variable being plotted
    ax.set_title("Circular City Index \nAutonomous Communities\nScheme = Quantiles", fontsize=20, y=1.01)
    # Remove axis clutter
    ax.set_axis_off()

    if SAVE_FIGS:
        plt.savefig(DIR_RESULTS + "map_results_CCI_ProvVsAutComm.svg", format="svg")

    # Display the figure
    plt.show()

    ### CCI Levels
    levels_variables = list(gdf_descriptive_main)[1:]

    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Start a loop over all the variables of interest
    for i, col in enumerate(levels_variables):
        # select the axis where the map will go
        ax = axs[i]
        # Plot the map
        gdf.plot(
            column=col,
            ax=ax,
            legend=True,
            legend_kwds={'loc':'lower right'},
            #edgecolor=line_color(AREA_TO_PREDICT),
            linewidth=0,
            figsize=(20, 20),
            cmap="Blues", #cmap="RdYlGn",
            scheme="NaturalBreaks", #scheme="Quantiles",
            k=5 #k=4
        )

        # Remove axis clutter
        ax.set_axis_off()
        if col == "Energy_Climate_Resources":
            # Set the axis title to the name of variable being plotted
            ax.set_title("Energy, Climate, and Resources (ECR)")

        else:
            # Set the axis title to the name of variable being plotted
            ax.set_title(col + " (" + col[0] + ")")

    if SAVE_FIGS:
        plt.savefig(DIR_RESULTS + "map_results_levels_All.svg", format="svg")

    # Display the figure
    plt.show()

    ### Key Performance Indicators (KPIs)
    kpi_variables = list(gdf_descriptive_KPIs)
    nrows_kpi = math.ceil(len(kpi_variables) / 2)

    f, axs = plt.subplots(nrows=nrows_kpi, ncols=2, figsize=(12, 12))

    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Start a loop over all the variables of interest
    for i, col in enumerate(kpi_variables):
        # select the axis where the map will go
        ax = axs[i]
        # Plot the map
        gdf.plot(
            column=col,
            ax=ax,
            edgecolor=line_color(AREA_TO_PREDICT),
            linewidth=0,
            cmap='Blues', # cmap='RdYlGn',
        )
        # Remove axis clutter
        ax.set_axis_off()
        # Set the axis title to the name of variable being plotted
        ax.set_title(col)

    if SAVE_FIGS:
        plt.savefig(DIR_RESULTS + "map_results_CCI_KPIs.svg", format="svg")

    # Display the figure
    plt.show()

    ### Highest scoring municipalities
    # highest municipality per level
    highest_MUN_per_level = pd.DataFrame(columns=['level', 'score','CTOT', 'Municipality', 'geometry'])
    target_columns = ['CCI','Digitalization', 'Energy_Climate_Resources', 'Mobility', 'Waste']

    for target in target_columns:
        level = target
        CTOT = gdf[target].idxmax()
        CTOT_str = str(CTOT)
        value = gdf[target].max()
        MUN_name = gdf['Municipality'].loc[CTOT_str]
        geometry = gdf['geometry'].loc[CTOT_str]

        new_row = {'level' : level,    
                    'score' : value,
                    'CTOT' : CTOT,
                    'Municipality':MUN_name,
                    'geometry' : geometry}

        highest_MUN_per_level = highest_MUN_per_level.append(new_row, ignore_index=True)

    if SAVE_TABLES is True:
        highest_MUN_per_level.to_csv((DIR_RESULTS + "descriptive_highest_MUN_per_level.csv"), index=False)

    highest_MUN_per_level


    # Define total highest 10 percent
    ten_percent = int(len(gdf) * 0.1)
    highest_CCI_list = gdf['CCI'].nlargest(n=ten_percent).index

    # highest municipality of CCI
    top_highest_MUN = pd.DataFrame(columns=['position_high', 'score','CTOT', 'Municipality', 'geometry'])

    n=1

    for CTOT in highest_CCI_list:
        level = 'CCI'
        Position = n
        value = gdf['CCI'].loc[CTOT]
        MUN_name = gdf['Municipality'].loc[CTOT]
        geometry = gdf['geometry'].loc[CTOT]

        new_row = {'position_high' : Position,    
                    'score' : value,
                    'CTOT' : CTOT,
                    'Municipality':MUN_name,
                    'geometry' : geometry}

        top_highest_MUN = top_highest_MUN.append(new_row, ignore_index=True)

        n=n+1

    top_highest_MUN

    # Read as GeoDataFrame
    gdf_top_highest = gpd.GeoDataFrame(top_highest_MUN)

    # Define CRS
    gdf_top_highest = gdf_top_highest.set_crs('epsg:4258')
    gdf_top_highest = gdf_top_highest.to_crs(gdf.crs)

    # Plot index results - Waste (W-level)
    fig, ax = plt.subplots(figsize=(20, 20))

    # Basemap
    gdf.plot(
        ax=ax, 
        alpha=0.4, 
        color="lightgrey",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    # GeoDataFrame with highest scoring municipalities
    gdf_top_highest.plot(
        ax=ax, 
        color="green",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    ax.set_title("Highest 10% scoring municipalities in CCI - " + AREA_TO_PREDICT , fontsize=20, y=1.01)

    #cx.add_basemap(ax, crs=gdf_top_highest.crs)

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "map_top_highest_MUN_CCI.svg", format="svg")

    plt.show()

    ### Lowest scoring municipalities
    # lowest municipality per level
    lowest_MUN_per_level = pd.DataFrame(columns=['level', 'score','CTOT', 'Municipality', 'geometry'])
    target_columns = ['CCI','Digitalization', 'Energy_Climate_Resources', 'Mobility', 'Waste']

    for target in target_columns:
        level = target
        CTOT = gdf[target].idxmin()
        CTOT_str = str(CTOT)
        value = gdf[target].min()
        MUN_name = gdf['Municipality'].loc[CTOT_str]
        geometry = gdf['geometry'].loc[CTOT_str]

        new_row = {'level' : level,    
                    'score' : value,
                    'CTOT' : CTOT,
                    'Municipality':MUN_name,
                    'geometry' : geometry}

        lowest_MUN_per_level = lowest_MUN_per_level.append(new_row, ignore_index=True)

    if SAVE_TABLES is True:
        lowest_MUN_per_level.to_csv((DIR_RESULTS + "descriptive_lowest_MUN_per_level.csv"), index=False)

    lowest_MUN_per_level

    # Define total lowest 10 percent
    ten_percent = int(len(gdf) * 0.1)
    lowest_CCI_list = gdf['CCI'].nsmallest(n=ten_percent).index

    # lowest municipality of CCI
    top_lowest_MUN = pd.DataFrame(columns=['position_low', 'score','CTOT', 'Municipality', 'geometry'])

    n=1

    for CTOT in lowest_CCI_list:
        level = 'CCI'
        Position = n
        value = gdf['CCI'].loc[CTOT]
        MUN_name = gdf['Municipality'].loc[CTOT]
        geometry = gdf['geometry'].loc[CTOT]

        new_row = {'position_low' : Position,    
                    'score' : value,
                    'CTOT' : CTOT,
                    'Municipality':MUN_name,
                    'geometry' : geometry}

        top_lowest_MUN = top_lowest_MUN.append(new_row, ignore_index=True)

        n=n+1

    top_lowest_MUN


    # Read as GeoDataFrame
    gdf_top_lowest = gpd.GeoDataFrame(top_lowest_MUN)

    # Define CRS
    gdf_top_lowest = gdf_top_lowest.set_crs('epsg:4258')
    gdf_top_lowest = gdf_top_lowest.to_crs(gdf.crs)

    # Plot index results - Waste (W-level)
    fig, ax = plt.subplots(figsize=(20, 20))

    # Basemap
    gdf.plot(
        ax=ax, 
        alpha=0.4, 
        color="lightgrey",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    # GeoDataFrame with lowest scoring municipalities
    gdf_top_lowest.plot(
        ax=ax, 
        color="red",
        edgecolor=line_color(AREA_TO_PREDICT),
        legend=True,
        )

    ax.set_title("Lowest 10% scoring municipalities in CCI - " + AREA_TO_PREDICT , fontsize=20, y=1.01)

    #cx.add_basemap(ax, crs=gdf_top_lowest.crs)

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "map_top_lowest_MUN_CCI.svg", format="svg")

    plt.show()

    ### Highest vs Lowest scoring municipalities
    sns.set_theme(style='white')
    # Plot index results - Waste (W-level)
    fig, ax = plt.subplots(figsize=(20, 20))

    # Basemap
    gdf.plot(
        ax=ax, 
        alpha=0.4, 
        color="lightgrey",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    # GeoDataFrame with 10 highest scoring municipalities
    gdf_top_highest.plot(
        ax=ax, 
        color="green",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    # GeoDataFrame with 10 lowest scoring municipalities
    gdf_top_lowest.plot(
        ax=ax, 
        color="red",
        edgecolor=line_color(AREA_TO_PREDICT)
        )

    ax.set_title("Highest 10% scoring municipalities vs lowest 10% scoring municipalities in CCI - " + AREA_TO_PREDICT , fontsize=20, y=1.01)
    ax.set_axis_off()
    #cx.add_basemap(ax, crs=gdf_lowest.crs)

    ax.legend(['First line', 'Second line'])

    if SAVE_FIGS is True:
        plt.savefig(DIR_RESULTS + "map_top_highestVSlowest_MUN_CCI.svg", format="svg")

    plt.show()

    print("The Descriptive Analysis of the Circular City Index are computed for {}".format(AREA_TO_PREDICT))
    print("##################################")
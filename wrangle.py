import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")
# importing my personal wrangle module
import wrangle


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.

    # I checked, and the following query should and better return exactly 52,442 rows
    sql_query = """
SELECT prop.*, 
       pred.logerror, 
       pred.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 

FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
"""
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in the zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df


def missing_values_col_wise(df):
    '''
    Function that takes in a df and returns a list of attributes with the number and percent
    of missing values for that attribute.
    '''
    new_df = pd.DataFrame(df.isnull().sum())
    new_df['number_missing'] = df.isnull().sum()
    new_df['percent_missing'] = df.isnull().sum()/len(df)*100
    new_df.drop([0], axis=1, inplace=True)
    return new_df


def missing_values_row_wise(df):
    '''
    Function that takes in a df and returns a list of rows with the number and percent
    of missing values.
    '''
    new_df = pd.DataFrame(df.isnull().sum(axis =1).value_counts())
    new_df['percent_missing'] = new_df.index/len(df.columns)*100
    new_df['num_of_rows'] = df.isnull().sum(axis =1).value_counts()
    new_df.drop([0], axis=1, inplace=True)
    new_df.index.rename('num_cols_missing_from_row', inplace = True)
    return new_df

def handle_missing_values(df, prop_required_row, prop_required_col):
    ''' 
    function which takes in a dataframe, proportion of non-null rows and columns
    (respectively) required to prevent the columns and rows being dropped:
    '''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


def remove_outliers(df, k, col_list):
    ''' 
    
    Here, we remove outliers from a list of columns in a dataframe and return that dataframe
    
    '''
    
    # My MVP will rely on the same outlier handling across rows; maybe I will get classier
    # about it in a later version
    
    # need a column list, either in or outside the function
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def clean_and_prep_data(df):
    '''
    This function will do some cleaning and other manipulation of the zillow data set, 
    to get it ready to be split in the next step.
    
    '''
    # Restrict df to only properties that meet single unit use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

        # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc'
       ,'regionidcounty', 'transactiondate', 'fips'
        ,'censustractandblock', 'propertylandusedesc', 'unitcnt'])


#     replace nulls in unitcnt with 1
#     df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
#     df.heatingorsystemdesc.fillna('None', inplace = True)

    # actually, I'm not assuming this, and I am dropping the heatingsystem col
    df.drop(columns = 'heatingorsystemdesc', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]

        # renaming columns for ease of reading
    df = df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms'
    ,'buildingqualitytypeid':'condition','calculatedfinishedsquarefeet':'sq_ft'
    ,'fullbathcnt':'full_baths','lotsizesquarefeet':'lot_size', 'rawcensustractandblock':'census_tract'
    ,'regionidcity':'city_id','regionidzip':'zip','roomcnt':'rooms','structuretaxvaluedollarcnt':'structure_value'
    ,'taxvaluedollarcnt':'tax_value','taxamount':'tax_amount'
    ,'assessmentyear':'year_assessed','landtaxvaluedollarcnt':'land_value'})

    cols = ['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount']

    #removing outliers--see the function elsewhere in this file
    df = remove_outliers(df, 1.5, cols)
        # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df,.7,.5)
    # accessing datetime info so as to create an 'age' variable
    from datetime import date
    df.yearbuilt =  df.yearbuilt#.astype(int)
    year = date.today().year
    df['age'] = year - df.yearbuilt
    # making a feature called bathrooms_per_sq_ft
    df['rooms'] = df.rooms.replace(to_replace=0.0,value=df.rooms.mean())
    df['sq_ft_per_bathroom'] = df.sq_ft / df.bathrooms
    # dropping the 'yearbuilt' column now that i have the age
    df['sq_ft_per_bedroom'] = df.sq_ft / df.bedrooms
    df['sq_ft_per_room'] = df.sq_ft / df.rooms
    df['has_half_bath'] = (df.bathrooms - df.full_baths) != 0
    df['has_half_bath'] = df.has_half_bath.astype(int)
    df['age_bin'] = pd.cut(df.age, [0,20,40,80,100,200], labels=[1,2,3,4,5])
    df = df.drop(columns=['yearbuilt'])
    df['tax_rate'] = df.tax_amount/df.tax_value*100
    df['price_per_sq_ft'] = df.structure_value/df.sq_ft

    # there were a few incorrect zip codes, <10, so i drop them here
    df = df[df.zip < 100_000]

    cols = ['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount', 'tax_rate','price_per_sq_ft']

    #removing outliers--see the function elsewhere in this file
    df = remove_outliers(df, 1.5, cols)

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()

    return df



def split_zillow(df):
    '''
    Takes in the zillow dataframe and returns train, validate, test subset dataframes
    '''
    # SPLIT
    # Test set is .2 of original dataframe
    train, test = train_test_split(df, test_size = .2, random_state=123)#, stratify= df.tax_value)
    # The remainder is here divided .7 to train and .3 to validate
    train, validate = train_test_split(train, test_size=.3, random_state=123)#, stratify= train.tax_value)

    # return train, validate, test

    X_train = train.drop(columns=['logerror'])
    # I was getting duplicate logerror columns, so the below code resolves by removing the duplicate
    y_train = train.logerror.T.drop_duplicates().T

    X_validate = validate.drop(columns=['logerror'])
    y_validate = validate.logerror.T.drop_duplicates().T

    X_test = test.drop(columns=['logerror'])
    y_test = test.logerror.T.drop_duplicates().T

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def encode_zillow(df):
    '''
    This is encoding a few of the zillow columns for later modelling; it drops the original column 
    once it has been encoded
    
    '''
    # ordinal encoder? sklearn.OrdinalEncoder

    # I had originally put the columns to be dummied inside the function. They are now in the inputs
    # cols_to_dummy = df['county']
    # df = pd.get_dummies(df, columns=['county'], dummy_na=False, drop_first=False)
        
    dummy_df = pd.get_dummies(df.county, dummy_na=False, drop_first=False)

    # Not requiring me to concatenate for some reason here.  
    df = pd.concat([df, dummy_df], axis = 1)
    df = df.rename(columns={'county_Los_Angeles':'Los_Angeles','county_Orange':'Orange','county_Ventura':'Ventura'})
    #df = df.drop(columns='county')
    return df



def scale_zillow(train, validate, test):
    '''
    Takes in the zillow dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # Create scaler; fit to train; transform onto different 
    scaler = MinMaxScaler()
    scaler.fit(train[['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath', 'tax_rate', 'price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura']])
    train_scaled = scaler.transform(train[['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura']])
    validate_scaled = scaler.transform(validate[['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura']])
    test_scaled = scaler.transform(test[['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura']])

    # Turn the datasets into pandas dataframes for further manipulation:
    train_scaled = pd.DataFrame(data=train_scaled, columns=['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura'])
    validate_scaled = pd.DataFrame(data=validate_scaled, columns=['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura'])
    test_scaled = pd.DataFrame(data=test_scaled, columns=['parcelid', 'bathrooms', 'bedrooms', 'condition', 'sq_ft', 'full_baths',
       'latitude', 'longitude', 'lot_size', 'census_tract', 'city_id', 'zip',
       'rooms', 'structure_value', 'tax_value', 'year_assessed', 'land_value',
       'tax_amount', 'logerror', 'age', 'sq_ft_per_bathroom',
       'sq_ft_per_bedroom', 'sq_ft_per_room', 'has_half_bath','tax_rate','price_per_sq_ft',
       'Los_Angeles', 'Orange', 'Ventura'])

    # Divide into x/y
    # I was getting duplicate columns in the y_ data sets, so I transposed, dropped duplicates, and transposed back
    X_train_scaled = train_scaled.drop(columns=['logerror'])
    y_train_scaled = train_scaled.logerror.T.drop_duplicates().T

    X_validate_scaled = validate_scaled.drop(columns=['logerror'])
    y_validate_scaled = validate_scaled.logerror.T.drop_duplicates().T

    X_test_scaled = test_scaled.drop(columns=['logerror'])
    y_test_scaled = test_scaled.logerror.T.drop_duplicates().T

    return train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled



def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    '''
    This function accesses (or creates) the zillow_df.csv file and wrangles it (clean, prep, split, scale)
    '''
    df = get_zillow_data()

    df = clean_and_prep_data(df)
    
    df = encode_zillow(df)

    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = split_zillow(df)

    train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled = scale_zillow(train, validate, test)

    return df, train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled

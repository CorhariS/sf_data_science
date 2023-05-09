import pandas as pd
import numpy as np
from statistics import median, mean
from sklearn.preprocessing import MinMaxScaler
import ast
import pickle


def process_homeFacts(facts, fact_labels):
    facts = eval(facts)
    homeFacts = {}
    for item in facts['atAGlanceFacts']:
        fact_labels.add(item['factLabel'])
        homeFacts[item['factLabel']] = item['factValue']    
    
    return homeFacts

def process_homeFacts_new(facts, fact_label):
    if fact_label in facts:
        return facts[fact_label]
    
    return None

def parsing_homeFacts(df):
    fact_labels = set()
    # Распарсим колонку homeFacts и запишим значения в новую колонку homeFacts_new в виде словаря {factLabel_1: factValue_1, factLabel_2: factValue_2, ...}
    df['homeFacts_new'] = df['homeFacts'].apply(lambda x: process_homeFacts(x, fact_labels))
    
    # Создаем в датасете новые колонки, имена которых содержатся во множестве fact_labels
    for item in fact_labels:
        df[item] = df['homeFacts_new'].apply(lambda x: process_homeFacts_new(x, item))
    # Удаляем колонки homeFacts и homeFacts_new
    df.drop(['homeFacts', 'homeFacts_new'], axis=1, inplace=True)
    return df



def process_schools(school):
    dict_schools = {'Elementary': 0,
                    'Middle': 0,
                    'High': 0,
                    'Other': 0, 
                    'Rating': 0, 
                    'Distance':0}
    school = ast.literal_eval(school)[0]
    rating = school['rating']
    distance = school['data']['Distance']
    names =  school['name']  #school['data']['Grades']
    lst_rating = []
    lst_distance = []
   
    for id_name, name in enumerate(names):
        if name in [None]:
            continue
        
        rating_value_str = rating[id_name].partition('/')[0]  # rating может принимать дробные значения (val/10), поэтому берем в рейтинг только чисслитель
        if rating_value_str in ['NR', 'NA', 'None'] :
            rating_value = 0
        else:
            rating_value = float(rating_value_str)             
             
        lst_rating.append(rating_value)
        
        distance_str = distance[id_name].partition('mi')[0]  # distance берем значения до 'mi'
        lst_distance.append(float(distance_str))  
        
        if 'Elementary' in name:
            dict_schools['Elementary'] += 1
        elif 'Middle' in name:
            dict_schools['Middle'] += 1
        elif 'High' in name:
            dict_schools['High'] +=1
        else:
            dict_schools['Other'] += 1    

    if len(lst_rating)!=0:
        dict_schools['Rating'] = mean(lst_rating)
    if len(lst_distance)!=0:
        dict_schools['Distance'] = median(lst_distance)
        
    return dict_schools      


def parsing_schools(df):
    df['schools_new'] = df['schools'].apply(process_schools)
    new_df = df['schools_new'].apply(pd.Series)
    # Удаляем колонки schools и schools_new
    df.drop(['schools', 'schools_new'], axis=1, inplace=True)
    return pd.concat([df,new_df],axis=1)


def process_year(year, dict_year):
       
    year = int(year)
    begin_year = year//10*10
    end_year = year//10*10 + 9
    
    if year>=2010:
        int_year = '>=2010'
    elif year < 1850:
        int_year = '<1850'
    else:
        int_year = dict_year[(begin_year,end_year)]
        
    return int_year


def parsing_year_build(df):
    lst = np.arange(1850,2020,10)
    dict_year = {}
    for ind, item in enumerate(lst):
        if ind==len(lst)-1:
            dict_year[(lst[ind],2023)] = f'>={lst[ind]})'
            break
        dict_year[(lst[ind],lst[ind+1]-1)] = f'({lst[ind]} - {lst[ind+1]-1})'
        
    df['interval_year'] = df['Year built'].apply(lambda x: process_year(x, dict_year))
    df.drop(['Year built'], axis=1, inplace=True)
    
    return df


######################## Подготовка файла для сервиса предсказания #########################

#Объявляем функцию, реализующую фильтрацию выбросов по методу z-отклонений
def outliers_z_score(data, feature, const_dict, log_scale=False):
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    
    feature_lower_bound = feature+'_lower_bound'
    feature_upper_bound = feature+'_upper_bound'
    lower_bound = const_dict[feature_lower_bound]
    upper_bound = const_dict[feature_upper_bound]
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]    
    
    return cleaned

def get_binary_value(value, ind, dict_bin, value_mode):
    
    if value in dict_bin.keys():
        return dict_bin[value][ind]
    else:
        return dict_bin[value_mode][ind]

def encode_binary_features(df, dict_binary, const_dict, lst_features):
    
    for feature in lst_features:
        dict_feature = dict_binary[feature]
        for ind, col_name in enumerate(dict_feature['col_names']):
            df[col_name] = df[feature].apply(lambda x: get_binary_value(x, ind, dict_feature, const_dict[feature+'_mode']))
    
    return df
    


def preparate_file(df, method_clean=False):
    
    # подгрузим константы и бинарный справочник

    with open('../libs/data/const_dict.pkl', 'rb') as pkl_file:
        const_dict = pickle.load(pkl_file)
    
    with open('../libs/data/dict_binary.pkl', 'rb') as pkl_file:
        dict_binary = pickle.load(pkl_file)
    
    df = df.drop(['status'], axis=1)
    
    # PrivatePool
    df['PrivatePool'] = df['PrivatePool'].fillna(df['private pool'])
    df['PrivatePool'] = df['PrivatePool'].apply(lambda x: 1 if x=='Yes' or x=='yes' else 0)

    # propertyType
    df['propertyType'] = df['propertyType'].fillna(const_dict['propertyType'])
    df['propertyType'] = df['propertyType'].str.lower()
    df['propertyType'] = df['propertyType'].apply(lambda x: x+' other')

    lst_propertyType = ['single', 'condo', ['land', 'lot'], ['townhouse','townhome'], 'multi', 'traditional', 
                        ['coop', 'co-op'], ['ranch', 'farm'], 'high rise', ['low-rise', 'low'], 'detached', 'mobile', ['contemporary','modern'],
                        ['1 story', '1', 'one story'], ['2 story', '2', '2 stories', 'two stories', 'two story', 'stories'], ['3 stor', '3'], ['colonial', 'transitional', 'historical'],
                        ['garden', 'cluster home'], 
                        ['craft', 'cottage', 'tri-level', 'bungalow', 'cape', 'spanish', 'mediterranean', 'victorian', 'florida', 'french', 'georgian', 'loft', 'art', 'tudor'],
                        ['other', 'custom', 'manufactured']]

    mask_nan = df['propertyType'].isna()

    for item in lst_propertyType:
        if type(item)==list:
            for ind, prprt in enumerate(item):
                df['propertyType'].where(~(df[~mask_nan].propertyType.str.contains(prprt) ), other=item[0], inplace=True)
        else:
            df['propertyType'].where(~(df[~mask_nan].propertyType.str.contains(item) ), other=item, inplace=True) 

    # Преобразуем колонку 'baths' 
    pattern = r'(\d*\.\d+|\d+)'
    df["baths"] = pd.to_numeric(df["baths"].str.extract(pattern)[0], downcast='float')
    df["baths"] = df["baths"].fillna(const_dict['baths'])         #df.groupby("zipcode")["baths"].transform("median")
    df["baths"] = df["baths"].fillna(0)
    #beds
    df["beds"] = pd.to_numeric(df["beds"].str.extract(pattern)[0], downcast='float')
    df["beds"] = df["beds"].fillna(const_dict['beds'])    # df.groupby("zipcode")["beds"].transform("median")
    df["beds"] = df["beds"].fillna(0)
    #fireplace
    df['fireplace'] = df['fireplace'].fillna('0')
    df['fireplace'] = df['fireplace'].apply(lambda x: 0 if x.lower() in ['0', 'no', 'not applicable'] else 1)
    # homeFacts и schools
    df = parsing_homeFacts(df)
    df = parsing_schools(df)
    #Year built    
    df['Year built'] = df['Year built'].replace('',np.nan)
    df['Year built'] = df['Year built'].replace('No Data',np.nan)
    pattern = r'(^\d{4}$)'
    df['Year built'] = df['Year built'].str.extract(pattern)[0]
    df['Year built'].fillna(const_dict['Year built'], inplace=True)    
    # парсим значения в колонке 'Year build' и записываем новые значения в колонку 'interval_year'
    df = parsing_year_build(df)
    # Remodeled year
    df['Remodeled year'] = df['Remodeled year'].str.extract(pattern)[0]
    df['Remodeled year'] = df['Remodeled year'].fillna(0)
    df['Remodeled year'] = df['Remodeled year'].apply(lambda x: 0 if x==0 else 1)
    # Price/sqft
    df['price/sqft'] = df['Price/sqft'].str.replace(',','')
    pattern = r'\b(\d+)\b'
    df['price/sqft'] = pd.to_numeric(df['price/sqft'].str.extract(pattern)[0])
    # df["price/sqft"] = df["price/sqft"].fillna(const_dict['price/sqft_group'])   
    df["price/sqft"] = df["price/sqft"].fillna(const_dict['price/sqft'])    
    # sqft
    df['sqft'] = df['sqft'].str.replace(',','')
    df['sqft'] = pd.to_numeric(df['sqft'].str.extract(pattern)[0])    
    # df["sqft"] = df["sqft"].fillna(const_dict['sqft_group'])       
    df["sqft"] = df["sqft"].fillna(const_dict['sqft']) 
    # Cooling
    df['Cooling'] = df['Cooling'].fillna('0') 
    df['Cooling'] = df['Cooling'].apply(lambda x: 0 if x.lower() in ['','no data', 'none'] else 1)
    # Heating
    df['Heating'] = df['Heating'].fillna('0') 
    df['Heating'] = df['Heating'].apply(lambda x: 0 if x.lower() in ['','no data', 'none'] else 1)
    # Parking
    df['Parking'] = df['Parking'].fillna('0')
    df['Parking'] = df['Parking'].apply(lambda x: 0 if x.lower() in ['','no data', 'none'] else 1)  
    
    if method_clean:
        mask25000 = df['sqft']<25000
        # Удалим записи, где значения параметра 'sqft' превышают 25000
        df = df[mask25000]
        mask3000 = df['price/sqft']<3000
        # Удалим записи, где значения параметра 'price/sqft' превышают 3000
        df = df[mask3000]
        # фильтрация выбросов по методу z-отклонений       
        df = outliers_z_score(df, 'sqft', const_dict, log_scale=True)
        df = outliers_z_score(df, 'price/sqft', const_dict, log_scale=True)

    # Кодируем категориальные признаки
    features = ['zipcode', 'city', 'state', 'propertyType', 'interval_year']
    df = encode_binary_features(df, dict_binary, const_dict, lst_features=features)

    # Удалим лишнии колонки
    df = df.drop(columns=['private pool',
                          'street', 
                          'stories',
                          'mls-id', 
                          'MlsId', 
                          'lotsize',
                          'Price/sqft',
                          'zipcode', 
                          'city', 
                          'state',
                          'propertyType',
                          'interval_year'                          
                          ], axis=1)
    
    
    # Проведем нормализацию следующих признаков, подгрузим MinMaxScaler с параметрами из const_dict
    col_scale = ['baths', 'sqft', 'beds', 'Elementary', 'Middle', 'High', 'Other', 'Rating', 'Distance', 'price/sqft']
    scaler = const_dict['MinMaxScaler_model']
    df[col_scale] = scaler.transform(df[col_scale].values)
    
    if method_clean:
        # в строках где отсутствует zipcode мы ничего не сможем предсказать, удалим эти строки
        df = df.dropna()
   
    return df[const_dict['order_columns_name']]





    


    
    
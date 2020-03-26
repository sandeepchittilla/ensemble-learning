"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Sonia Kamaal
        (2) Sai Sandeep Chittilla
        (3) Taneja Ankisetty
"""

"""
    Import necessary packages
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import string
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
stemmer = PorterStemmer()
stemmer_fr = FrenchStemmer()
import os.path
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

# pip3 install quantulum3
from quantulum3 import parser

# pip install --upgrade scikit-learn
from sklearn.ensemble import StackingClassifier

# Dependencies for lightgbm
# pip install lightgbm
# brew install cmake
# brew install libomp
import lightgbm as lgb 

# Installation spacy packages depends on the confirguration of laptop. So one of the below works
# spacy_nlp_fr = spacy.load('fr_core_news_md')
# spacy_nlp_en = spacy.load('en_core_web_md')
spacy_nlp_fr = spacy.load('fr_core_news_sm')
spacy_nlp_en = spacy.load('en_core_web_sm')

# pip install regex
import regex as re


print('########## SUCCESSFULLY INSTALLED PACKAGES ########## ')

# Preprocessing Steps
def remove_html_tags(text): 
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

def convert_html(string):
    """ Replaces html tags """
    string = string.replace('&Agrave;','À')
    string = string.replace('&agrave;','à')
    string = string.replace('&Acirc;','Â')
    string = string.replace('&acirc;','â')
    string = string.replace('&AElig;','Æ')
    string = string.replace('&aelig;','æ')
    string = string.replace('&Ccedil;','Ç')
    string = string.replace('&ccedil;','ç')
    string = string.replace('&Egrave;','È')
    string = string.replace('&egrave;','è')
    string = string.replace('&Eacute;','É')
    string = string.replace('&eacute;','é')
    string = string.replace('&Ecirc;','Ê')
    string = string.replace('&ecirc;','ê')
    string = string.replace('&Euml;','Ë')
    string = string.replace('&euml;','ë')
    string = string.replace('&Icirc;','Î')
    string = string.replace('&icirc;','î')
    string = string.replace('&Iuml;','Ï')
    string = string.replace('&iuml;','ï')
    string = string.replace('&Ocirc;','Ô')
    string = string.replace('&ocirc;','ô')
    string = string.replace('&OElig;','Œ')
    string = string.replace('&oelig;','œ')
    string = string.replace('&Ugrave;','Ù')
    string = string.replace('&ugrave;','ù')
    string = string.replace('&Ucirc;','Û')
    string = string.replace('&ucirc;','û')
    string = string.replace('&Uuml;','Ü')
    string = string.replace('&uuml;','ü')
    string = string.replace('&laquo;','«')
    string = string.replace('&raquo;','»')
    string = string.replace('&euro;','€')
    string = string.replace('&bull;',' ') # removing bullet points
    string = string.replace('&rsquo;',"'")

    return string

# converting ascii characters
def convert_ascii(string):
#     string = string.replace('&#39;', "'")
    string = re.sub('&#[0-9]+',r' ', string)
    string = string.replace('\xa0',' ')
    return string

# removing special characters
def remove_special(string):
    string = string.replace('.nan',' ')
    string = string.replace('®',' ')
    string = string.replace('N°',' ')
    string = ''.join([i if ord(i) < 128 else ' ' for i in string])
    return string

def splitAlphaNumeric(string):
    string = re.sub('([0-9]+)([aA-zZ]+)',r'\g<1> \g<2>', string)
    string = re.sub('([aA-zZ]+)([0-9]+)',r'\g<1> \g<2>', string)
    return string

def replaceMetrics(string):
    string = string.replace('m³',' m3 ')
    string = string.replace('m²', ' m2 ')
    string = string.replace('m³la', ' m3 la')
    string = string.replace('m³kitprov', ' m3 kitprov')
    string = string.replace('m³kit', ' m3 kit')
    string.replace('cm²',' cm2 ')
    string.replace('cm³',' cm3 ')
    string.replace('µ',' mu ')
    string.replace('gm²',' gm2 ')
    string.replace('ma²',' ma2 ')
    string.replace('ß',' beta ')
    string.replace('mm²',' mm2 ')
    string.replace('m²',' m2 ')
    return string

def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')
    string = string.replace('à', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    string = string.replace('ñ', 'n')
    
    string.replace('ª','a')
    string.replace('ä','a')
    string.replace('ã','a')
    string.replace('ø','o')
    string.replace('œ','ae')
    string.replace('º',' ')
    
    return string

def raw_to_tokens(raw_string):

    string = raw_string.lower()
    string = normalize_accent(string)
    
    # french lemma
    spacy_tokens = spacy_nlp_fr(string)
    string_tokens = [token.lemma_ and token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    string = " ".join(string_tokens)
    
    # en lemma
    spacy_tokens = spacy_nlp_en(string)
    string_tokens = [token.lemma_ and token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    string = " ".join(string_tokens)
    
    string = " ".join([stemmer.stem(w.lstrip('0')) for w in string.split()]) # en stemmer and removing leading 0's
    clean_string = " ".join([stemmer_fr.stem(w) for w in string.split()]) # fr stemmer
        
    return clean_string


# Feature Engineering
# Identify if there is any year (eg. 1995, 20110) in the designation column
def get_year_info(all_data):
    year = []
    for i in all_data:
        year1 = re.match(r'.*([^\d][1][8-9][\d]{2}[^\d])', str(i))
        year2 = re.match(r'.*([^\d][2][0][1|2][\d][^\d])', str(i))
        if year1 or year2 is not None:
            year.append(1)
        else:
            year.append(0)
    return(year)

# Identify if there is any dimension (eg. 10x10, 5-5) in the designation column
def get_dim_info(all_data):
    dimensions = []
    err_count=0
    err_rec=[]
    for i in all_data:
        try:
            dim_list = re.search(r'(?<!\S)\d+(?:,\d+)? ?x ?\d+(?:,\d+)?(?: ?x ?\d+(?:,\d+)?)*', i)
        except:
            dim_list = re.search(r'(?<!\S)\d+(?:,\d+)? ?x ?\d+(?:,\d+)?(?: ?x ?\d+(?:,\d+)?)*', str(i))
            err_count+=1
            err_rec.append(i)
        if dim_list is not None:
            dimensions.append(1)
        else:
            dimensions.append(0)
    print(err_count)
    print(err_rec)
    return(dimensions)

# Identify if there is any color (eg. red, noir) in the designation column
def get_colour_info(all_data):
    colours=[]
    for i in all_data:
        colour_list = re.search(r'(\L<colours>)', str(i), colours = [
        'Red', 'rouge', 'yellow', 'jaune', 'blue', 'bleu', 'bleue' 'green', 'vert',
        'verte', 'orange', 'white', 'blanc', 'blanche', 'black', 'noir', 'noire', 'gray',
        'gris', 'grise', 'brown', 'marron', 'pink', 'rose', 'purple', 'violet', 'violette'])
        if colour_list is not None:
            colours.append(1)
        else:
            colours.append(0)
    return(colours)


# Identify if there is any measurement units (eg. m2, cm3) in the designation column
def get_quants_info(all_data):
	quants=[]
	for i in all_data:
		try:
			quants_list = parser.parse(i)
			if(str(quants_list[0].unit.name) == 'dimensionless'):
				quants.append(0)
			else:
				try:
					quants.append(1)
				except Exception as e: # catches issues when there is no quant term detected
					quants.append(0)
		except: # cactches issues with the parsing itself  (should be rare)
			quants.append(0)
	return(quants)

# Captures length of the designation string as a feature
def mapCharLength(string):
    char_len = len(str(string))
    
    if char_len<50:
        return 0
    elif char_len<100:
        return 1
    elif char_len<150:
        return 2
    elif char_len<200:
        return 3
    else:
        return 4


# Encoding y_train labels
def mapPrdIdtoInteger(df, col_name):
    uniqueProdIds = [i for i in pd.unique(df[col_name].values)]
    prdIdDict = {}
    for i in range(len(uniqueProdIds)):
        prdIdDict.setdefault(uniqueProdIds[i],i)
    return prdIdDict



# #SUBSAMPLING
# req_indices = y_train['rank']<=1000
# x_train_limit = x_train_svd_all[req_indices]
# y_train_limit = y_train[req_indices]
# x_train_limit.shape



"""
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""

"""
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
"""

def model_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators = 100, max_depth=2, random_state=0) # please choose all necessary parameters
    
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def model_xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(max_depth=3, min_child_weight=1, n_estimators=100)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    xg_accuracy = f1_score(y_test, y_pred, average = 'weighted')
    xg_f1 = accuracy_score(y_test, y_pred)
    
    return xg_accuracy, xg_f1


def model_adaboost(X_train, y_train, X_test, y_test):
    clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    ada_accuracy = f1_score(y_test, y_pred, average = 'weighted')
    ada_f1 = accuracy_score(y_test, y_pred)

    return ada_accuracy, ada_f1

def model_decisiontree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(max_features='sqrt',random_state=7,)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    dt_accuracy = f1_score(y_test, y_pred, average = 'weighted')
    dt_f1 = accuracy_score(y_test, y_pred)

    return dt_accuracy, dt_f1


def model_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(min_child_samples=20, min_child_weight=0.001, n_estimators=100, num_leaves=31)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lgb_accuracy = f1_score(y_test, y_pred, average = 'weighted')
    lgb_f1 = accuracy_score(y_test, y_pred)

    return lgb_accuracy, lgb_f1


def model_stack(X_train, y_train, X_test, y_test):
	estimators =  [('xgb', XGBClassifier()), ('lgb', lgb.LGBMClassifier())]
	model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	stack_accuracy = f1_score(y_test, y_pred, average = 'weighted')
	stack_f1 = accuracy_score(y_test, y_pred)

	return stack_accuracy, stack_f1

def model_train_lgb(X_train, y_train, X_test, y_test):
    params = {
        "objective" : "multiclass",
        "metric" : "",
        "learning_rate" : 0.01,
        "num_class" : 27
    }
    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_test, label=y_test)
    evals_result = {}
    model = lgb.train(params=params, train_set=lgtrain, num_boost_round=150, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=60, 
                      verbose_eval=50,
                      evals_result=evals_result)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    predictions = []
    for x in y_pred:
        predictions.append(np.argmax(x))

    train_lgb_accuracy = f1_score(y_test, predictions, average = 'weighted')
    train_lgb_f1 = accuracy_score(y_test, predictions)
    return train_lgb_accuracy, train_lgb_f1



"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":


    # Import Train and Test data
    x_train = pd.read_csv('X_train.csv',header = 0, names=['IntegerId','Designation','Description','ProductId','ImageId'])
    # x_test = pd.read_csv('X_test.csv',header = 0, names=['IntegerId','Designation','Description','ProductId','ImageId'])
    y_train = pd.read_csv('Y_train.csv', header =0, names = ['IntegerId','PrdTypeCode'])

    # Merge Designation and Description columns
    x_train['Merged Des'] = x_train['Designation'].astype(str) + ' ' + x_train['Description'].astype(str)
    # x_test['Merged Des'] = x_test['Designation'].astype(str) + ' ' + x_test['Description'].astype(str)

    if not os.path.isfile('x_train_cleaned.csv'):
        # Clean Designation column of train
        clean = []
        raw = list(x_train['Merged Des'])
        for i in tqdm(raw):
            text = convert_ascii(i)
            text = remove_html_tags(text)
            text = convert_html(text)
            text = remove_special(text)
            text = splitAlphaNumeric(text)
            text = replaceMetrics(text)
            
            clean.append(raw_to_tokens(text))
            

        # Add the cleaned designation column back to the dataframe
        x_train['CleanDesignation'] = clean

        x_train.to_csv('x_train_cleaned.csv')
    x_train = pd.read_csv('x_train_cleaned.csv')

    # Imputations of Null values
    # There are only 10 values, we can remove them from the training.
    # However, if they are present in test data, we can directly predict them to be of 2705 PrdTypeCode
    req_indices = ~ x_train['CleanDesignation'].isnull()
    x_train = x_train[req_indices]
    y_train = y_train[req_indices]

    # Including all the pre-processing steps to x_train dataset
    x_train['Year'] = get_year_info(x_train['CleanDesignation'])
    x_train['Dimensions'] = get_dim_info(x_train['CleanDesignation'])
    x_train['Colour'] = get_colour_info(x_train['CleanDesignation'])
    x_train['Quants'] = get_quants_info(x_train['CleanDesignation'])
    x_train['char_length_class'] = x_train['CleanDesignation'].apply(lambda x: mapCharLength(x))

    print('########## SUCCESSFULLY FINISHED PRE-PROCESSING ########## ')

    # saving and reading the file to avoid loss of information
    x_train.to_csv('x_train_processed.csv')
    x_train = pd.read_csv('x_train_processed.csv',index_col=0)

    # tf-idf vectorizer on cleandesignation column which has all the pre-processing steps
    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.9)
    x_train_tfidf = tfidfconverter.fit_transform(list(x_train['CleanDesignation'])).toarray()
    x_train_tfidf_all = pd.DataFrame(x_train_tfidf,columns=tfidfconverter.get_feature_names())


    # In order to reduce number of features, we did dimension reduction using SVD
    if not os.path.isfile('train4k.pkl'):
    	svd = TruncatedSVD(n_components = 4000)
    	svd.fit(x_train_tfidf_all)
    	print('Variance explained after SVD',svd.explained_variance_ratio_.sum())

    	x_train_svd = svd.transform(x_train_tfidf_all)
    	print("Shape of X_train after SVD",x_train_svd.shape)

    	feature_names = tfidfconverter.get_feature_names()
    	best_features = [feature_names[i] for i in svd.components_[0].argsort()[::-1]]
    	best_features = best_features[0:svd.n_components]
    	x_train_svd_all = pd.DataFrame(x_train_svd,columns=best_features)

    	# Save the updated x_train file to prevent loss of information
    	x_train_svd_all.to_pickle("train4k.pkl")
    x_train_svd_all = pd.read_pickle("train4k.pkl")

    print('########## SUCCESSFULLY FINISHED TF-IDF AND SVD ########## ')

    # Adding feature engineering steps to the output of SVD
    x_train_svd_all['Year'] = x_train['Year']
    x_train_svd_all['Dimensions'] = x_train['Dimensions']
    x_train_svd_all['Colour'] = x_train['Colour']
    x_train_svd_all['Quants'] = x_train['Quants']
    x_train_svd_all['char_length_class'] = x_train['char_length_class']

    # Dropping the null values post SVD
    x_train_svd_all = x_train_svd_all.drop(17433)
    x_train_svd_all = x_train_svd_all.drop(9260)
    y_train = y_train.drop(y_train.index[[17433,9260]])

    # Reset index to match
    y_train.reset_index(inplace=True)
    x_train_svd_all.reset_index(inplace=True)

    y_train_df = pd.DataFrame(y_train['PrdTypeCode'])
    dictionary = mapPrdIdtoInteger(y_train_df,'PrdTypeCode')
    y_train = pd.DataFrame.replace(y_train_df, to_replace = dictionary)

    # Split the data to train data and validation data
    x_train_cv, x_val, y_train_cv, y_val = train_test_split(x_train_svd_all, y_train, test_size = 0.2)

    print('########## STARTING TO RUN EACH OF THE MODELS ########## ')

    print('########## STARTING TRAINED LIGHTGBM ########## ')
    train_lgb_accuracy, train_lgb_f1 = model_train_lgb(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING DECISION TREES ########## ')
    dt_accuracy, dt_f1 = model_decisiontree(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING RANDOM FORESTS ########## ')
    rf_accuracy, rf_f1 = model_random_forest(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING ADABOOST ########## ')
    ada_accuracy, ada_f1 = model_adaboost(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING XGBOOST ########## ')
    xg_accuracy, xg_f1 = model_xgboost(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING LIGHTGBM ########## ')
    lgb_accuracy, lgb_f1 = model_lightgbm(x_train_cv, y_train_cv, x_val, y_val)

    print('########## STARTING STACKING CLASSIFIER ########## ')
    stack_accuracy, stack_f1 = model_stack(x_train_cv, y_train_cv, x_val, y_val)

    # print the results
    print("resutls from Random Forest: accuracy = {} and f1 score = {}".format(rf_accuracy, rf_f1))
    print("resutls from XGBosst: accuracy = {} and f1 score = {}".format(xg_accuracy, xg_f1))
    print("resutls from AdaBoost: accuracy = {} and f1 score = {}".format(ada_accuracy, ada_f1))
    print("resutls from Decision Trees: accuracy = {} and f1 score = {}".format(dt_accuracy, dt_f1))
    print("resutls from LightGBM: accuracy = {} and f1 score = {}".format(lgb_accuracy, lgb_f1))
    print("resutls from Stacking Classifier: accuracy = {} and f1 score = {}".format(stack_accuracy, stack_f1))
    print("resutls from Trained LightGBM: accuracy = {} and f1 score = {}".format(train_lgb_accuracy, train_lgb_f1))




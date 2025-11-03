
import logging as log

import argparse
import os
import glob
from csv import DictReader
import pickle
import warnings

import random

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def read_data( corpus_root_dir ):
    ids = []
    labels = []
    examples = []
    
    ## TODO
    
    return ids , labels , examples

def read_file( corpus_file ):
    ids = []
    labels = []
    examples = []
    
    with open( corpus_file , 'r' ) as fp:
        csv_dict_reader = DictReader( fp , delimiter = '\t' )
        for cols in csv_dict_reader:
            content_id = cols[ 'id' ]
            ids.append( content_id )
            label = int( cols[ 'label' ] )
            labels.append( label )
            content = cols[ 'content' ]
            examples.append( re.sub(r'\d+', '', content ) )
    return ids , labels , examples


if __name__ == '__main__':

    ##warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()

    parser.add_argument( '-v' , '--verbose' ,
                         help = "print more information" ,
                         action = "store_true" )
    
    ## models
    parser.add_argument( "--model-file" ,
                         dest = 'model_file' ,
                        default = "./models/model.sav" ,
                        type = str ,
                        help = 'saved model' )
    parser.add_argument( "--feature-file" ,
                         dest = 'feature_file' ,
                         default = None ,
                         type = str ,
                         help = 'Plain text list of features' )
    parser.add_argument( "--max-features" ,
                         dest = 'max_features' ,
                         default = 1000 ,
                         type = int ,
                         help = 'Maximum feature size (If feature file provided, then that takes precedence)' )
    parser.add_argument( "--kernel" ,
                         dest = 'kernel' ,
                         default = 'linear' ,
                         type = str ,
                         help = 'SVM kernel type' )
    parser.add_argument( "--vectorizer-file" ,
                         dest = 'vectorizer_file' ,
                         default = "./models/vectorizer.sav" ,
                         type = str ,
                         help = 'saved tf-idf vectorizer' )
    parser.add_argument( '--overwrite-vectorizer' ,
                         dest = 'overwrite_vectorizer' ,
                         help = "When present, vectorizer file will be overwritten.  Otherwise, existing vectorizer will be used" ,
                         action = "store_true" )
    ## data
    parser.add_argument( "--data-dir" ,
                         dest = 'data_dir' ,
                         default = None ,
                         type = str ,
                         help = 'directory containing note data' )
    parser.add_argument( "--data-file" ,
                         dest = 'data_file' ,
                         default = None ,
                         type = str ,
                         help = 'Tab-delimited file containing note data and label' )
    ## output
    parser.add_argument( "--output-file" ,
                         dest = 'output_file' ,
                         default = None ,
                         type = str ,
                         help = 'Tab-delimited file containing validation data output predictions and reference labels' )
    ## ####
    args = parser.parse_args()
    ## ####
    ## Set up logging
    if args.verbose:
        log.basicConfig( format = "%(levelname)s: %(message)s" ,
                         level = log.DEBUG )
        log.info( "Verbose output." )
        log.debug( "{}".format( args ) )
    else:
        log.basicConfig( format="%(levelname)s: %(message)s" )

    ## ########
    if( args.output_file != None ):
        output_dir = os.path.dirname( args.output_file )
        if( not os.path.exists( output_dir ) ):
            log.warning( 'Creating output folder because it does not exist:  {}'.format( output_dir ) )
            try:
                os.makedirs( output_dir )
            except OSError as e:
                log.error( 'OSError caught while trying to create reference output folder:  {}'.format( e ) )
            except IOError as e:
                log.error( 'IOError caught while trying to create reference output folder:  {}'.format( e ) )
        else:
            log.warning( '{}'.format( output_dir ) )
    
    ## ########
    if( args.data_file != None ):
        ids , labels , corpus = read_file( args.data_file )
    elif( args.data_dir != None ):
        ids , labels , corpus = read_data( args.data_dir )
        
    ## ########
    x_train , x_valid, y_train , y_valid = train_test_split( corpus , labels ,
                                                             test_size = 0.25 ,
                                                             random_state = 213 )

    ## ####
    log.info( 'Train Size = {}'.format( len( x_train ) ) )
    log.info( 'Valid Size = {}'.format( len( x_valid ) ) )

    vocab = None
    max_features = args.max_features
    if( args.feature_file != None ):
        vocab = []
        with open( args.feature_file , 'r' ) as fp:
            for line in fp:
                line = line.strip()
                vocab.append( line )
        max_features = len( vocab )                
    ## ################################    
    if( args.overwrite_vectorizer or
        not os.path.exists( args.vectorizer_file ) ):
        ## ########
        vectorizer = TfidfVectorizer( analyzer = 'word' ,
                                      binary = False ,
                                      decode_error = 'strict' ,
                                      encoding = 'utf-8' ,
                                      input = 'content' ,
                                      lowercase = True ,
                                      max_df = 1.0 ,
                                      max_features = max_features ,                                      
                                      min_df = 1 ,
                                      ngram_range = ( 1 , 2 ) ,
                                      norm = 'l2' ,
                                      preprocessor = None ,
                                      smooth_idf = True ,
                                      stop_words = 'english' ,
                                      strip_accents = 'unicode' ,
                                      sublinear_tf = False ,
                                      token_pattern = "(?u)\\b\\w\\w+\\b" ,
                                      tokenizer = None ,
                                      use_idf = True ,
                                      vocabulary = vocab )
        ## #####
        x_train_vec = vectorizer.fit_transform( x_train )
        with open( args.vectorizer_file , 'wb' ) as fp:
            pickle.dump( vectorizer , fp )
    else:
        with open( args.vectorizer_file , 'rb' ) as fp:
            vectorizer = pickle.load( fp )
        ## ####
        x_train_vec = vectorizer.transform( x_train )
    ## ########
    x_valid_vec = vectorizer.transform( x_valid )
    
    ## ################################
    ## grid search
    search_space = { "C" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] }
    ##search_target = 'roc_auc'
    search_target = 'recall'
    cv = 5

    ## ################
    log.info( 'GridSearchCV-ing validation set ({}-fold) for optimal params'.format( cv ) )
    model = svm.SVC( kernel = args.kernel ,
                     probability = True ,
                     random_state = 213 )
    grid_search = GridSearchCV( estimator = model ,
                                param_grid = search_space ,
                                scoring = search_target ,
                                cv = cv ,
                                n_jobs = -1 )
    grid_search.fit( x_valid_vec , y_valid )
    log.info( 'The best hyper-parameters are:  {}'.format( grid_search.best_params_ ) )
    model.set_params( **grid_search.best_params_ )
         
    ## ########
    log.info( 'Fitting model.' )
    model.fit( x_train_vec , y_train )
    with open( args.model_file , 'wb' ) as fp:
        pickle.dump( model , fp )
    
    ## ################################
    log.info( 'Generating validation scores.' )
    y_train_labels = model.predict( x_train_vec )
    y_train_probs = model.predict_proba( x_train_vec )[:, 1]
    ## ########
    print( "Train report: " + classification_report( y_train , y_train_labels ) )
    print( 'Training AUC: {:0.4f}'.format( roc_auc_score( y_true = y_train ,
                                                          y_score = y_train_probs ) ) )
    print( "Train completed." )

    ## ########
    y_valid_labels = model.predict( x_valid_vec )
    y_valid_probs = model.predict_proba( x_valid_vec )[:, 1]
    ## ########
    print( "Validation report: " + classification_report( y_valid , y_valid_labels ) )
    print( 'Validation AUC:    {:0.4f}'.format( roc_auc_score( y_true = y_valid ,
                                                               y_score = y_valid_probs ) ) )
    print( "Validation completed." )
    ## ########
    if( args.output_file != None ):
        log.info( 'Writing validation output and probs to file.' )
        with open( args.output_file , 'w' ) as fp:
            fp.write( '{}\t{}\t{}\t{}\n'.format( 'id' ,
                                                 'RefLabel' ,
                                                 'ModelLabel' ,
                                                 'ModelProb' ) )
            for i , label in enumerate( y_valid ):
                fp.write( '{}\t{}\t{}\t{}\n'.format( ids[ i ] ,
                                                     label ,
                                                     y_valid_labels[ i ] ,
                                                     y_valid_probs[ i ] ) )


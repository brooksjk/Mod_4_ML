
import logging as log

import argparse
import pickle
import warnings
import json

from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':

    ##warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()

    ## models
    parser.add_argument("--vectorizer-file" ,
                        dest = 'vectorizer_file' ,
                        default = "./models/text_vectorizer.sav" ,
                        type = str ,
                        help = 'saved tf-idf vectorizer' )
    parser.add_argument("--feature-file" ,
                        dest = 'feature_file' ,
                        default = None ,
                        ##default = "./models/text_features.txt" ,
                        type = str ,
                        help = 'Plain text file for writing extracted features to' )
    parser.add_argument("--param-file" ,
                        dest = 'param_file' ,
                        default = None ,
                        ##default = "./models/text_params.json" ,
                        type = str ,
                        help = 'JSON text file for writing vectorizer params to to' )
    ## ####
    args = parser.parse_args()
    ## ####

    vectorizer = None
    with open( args.vectorizer_file , 'rb' ) as fp:
        vectorizer = pickle.load( fp )

    ## ################################
    ##
    if( args.feature_file != None ):
        try:
            features = vectorizer.get_feature_names_out()
        except AttributeError as e:
            log.warning( 'This appears to be an older version of scikit-learn. Consider upgrading.' )
            features = vectorizer.get_feature_names()
        with open( args.feature_file , 'w' ) as fp:
            for feat in features:
                fp.write( '{}\n'.format( feat ) )
    ## ################################
    if( args.param_file != None ):
        params = vectorizer.get_params()
        ## We need to drop 'dtype' since it isn't safe to convert to
        ## JSON
        params.pop( 'dtype' )
        params_json = json.dumps( params )
        with open( args.param_file , 'w' ) as fp:
            json.dump( params , fp , indent = 4 )
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging as log

import argparse
import os
import pickle
import glob
from csv import DictReader
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score


def read_data( corpus , output_dir ):
    ids = []
    examples = []
    
    with open( os.path.join( args.output_dir ,
                             'filenames.txt' ) , 'w' ) as out_fp:
        for file_name in glob.glob( os.path.join( corpus , '*.txt' ) ):
            with open( file_name , 'r') as f:
                ids.append( file_name )
                data = f.read()
                examples.append( re.sub(r'\d+', '', data ) )
                out_fp.write( '{}\n'.format( file_name ) )
        
    return ids , examples


def read_file( corpus_file , output_dir ):
    ids = []
    labels = []
    examples = []
    
    with open( os.path.join( args.output_dir ,
                             'filenames.txt' ) , 'w' ) as out_fp:
        with open( corpus_file , 'r' ) as fp:
            csv_dict_reader = DictReader( fp , delimiter = '\t' )
            for cols in csv_dict_reader:
                content_id = cols[ 'id' ]
                ids.append( content_id )
                if( 'label' in cols ):
                    label = int( cols[ 'label' ] )
                    labels.append( label )
                content = cols[ 'content' ]
                examples.append( content )
                ## ########
                out_fp.write( '{}\n'.format( content_id ) )
    ## ########
    return ids , labels , examples


if __name__ == '__main__':

    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()

    parser.add_argument( '-v' , '--verbose' ,
                         help = "print more information" ,
                         action = "store_true" )
    
    # models

    parser.add_argument( "--model-file" ,
                         default = "./models/cui_model.sav" ,
                         type = str ,
                         help = 'pre-trained saved model' )
    parser.add_argument( "--vectorizer-file" ,
                         default = "./models/cui_vectorizer.sav" ,
                         type = str ,
                         help = 'saved tf-idf vectorizer' )
    
    # data

    parser.add_argument("--cui_data_dir", default=None, type=str,
                        help='directory contains cui data')
    parser.add_argument("--text_data_dir", default=None, type=str,
                        help='directory contains text data')
    parser.add_argument( "--data-file" ,
                         dest = 'data_file' ,
                         default = None ,
                         type = str ,
                         help = 'Tab-delimited file containing note data and label' )
    
    # output dir
    ## TODO - support the output directory being option
    parser.add_argument( "--output-dir" ,
                         default = None ,
                         required = True ,
                         type = str ,
                         help = 'directory to store predictions' )
    parser.add_argument( "--output-file",
                         dest = 'output_file' ,
                         default = None ,
                         type = str ,
                         help = 'Tab-delimited file containing test data output predictions and reference labels' )
    
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
    
    # read data
    log.info( '***Loading data***' )
    examples = None
    labels = []
    if( args.data_file != None ):
        ids , labels , examples = read_file( args.data_file , args.output_dir )
    elif args.cui_data_dir is not None:
        ids , examples = read_data( args.cui_data_dir , args.output_dir )
    else:
        ids , examples = read_data( args.text_data_dir , args.output_dir )
    log.info( 'Finished loading notes.' )
    log.info( 'Number of examples = {}'.format( len(examples) ) )
    ## ########
    log.info( '\n\n***Loading model and vectorizer***' )
    with open( args.vectorizer_file , 'rb' ) as f:
        vectorizer = pickle.load( f )
    with open( args.model_file , 'rb' ) as f:
        model = pickle.load( f )
    log.info( 'Finished loading.' )
    
    log.info( '\n\n***Start predicting***' )
    x = vectorizer.transform(examples)
    predicted_labels = model.predict(x)
    predicted_probs = model.predict_proba(x)[:, 1]
    log.info( 'Finished predicting.' )
    ## ########
    if( labels != [] ):
        log.info( 'Generating report.' )
        print( "Testing report: " + classification_report( labels , predicted_labels ) )
        print( 'Testing AUC:    {:0.4f}'.format( roc_auc_score( y_true = labels ,
                                                                y_score = predicted_probs ) ) )
    
    ## ########
    log.info( 'Writing output, probabilities, and labels to file.' )
    
    with open(os.path.join( args.output_dir , 'predicted_labels.txt'), 'w') as f:
        for i in predicted_labels.tolist():
            f.write(str(i))
            f.write('\n')

    with open(os.path.join( args.output_dir , 'predicted_probabilities.txt'), 'w') as f:
        for i in predicted_probs.tolist():
            f.write(str(i))
            f.write('\n')
    ## ########
    if( args.output_file != None ):
        with open( args.output_file , 'w' ) as fp:
            fp.write( '{}\t{}\t{}\t{}\n'.format( 'id' ,
                                                 'RefLabel' ,
                                                 'ModelLabel' ,
                                                 'ModelProb' ) )
            for i , filename in enumerate( ids ):
                ## TODO - this is slower to generate but very easy to
                ## understand what is going on.  We can make it more
                ## efficient.  When there aren't any provided
                ## reference labels for this test corpus, then the
                ## labels data structure will be empty and we should
                ## print the empty string to the RefLabel field.
                ## Otherwise (when there were labels provided with the
                ## corpus), we should add the original RefLabel to
                ## this output CSV file.
                if( len( labels ) == 0 ):
                    this_label = ''
                else:
                    this_label = labels[ i ]
                fp.write( '{}\t{}\t{}\t{}\n'.format( filename ,
                                                     this_label ,
                                                     predicted_labels[ i ] ,
                                                     predicted_probs[ i ] ) )
    log.info( 'Finished saving' )

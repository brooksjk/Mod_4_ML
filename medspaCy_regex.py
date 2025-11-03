import argparse
import glob
import sys
import os
import re
import pandas as pd
from tqdm import tqdm

import spacy
import medspacy
from quickumls.constants import MEDSPACY_DEFAULT_SPAN_GROUP_NAME
from medspacy.ner import TargetRule

import cassis

def main( inputDir , outputDir , typesDir ,
          engine ,
          verboseFlag = False ):
    nlp_pipeline = None
    if( engine in [ 'en_core_web_sm' ] ):
        ## https://github.com/medspacy/medspacy/blob/master/notebooks/03-Information-Extraction.ipynb
        nlp_pipeline = spacy.load( engine )
    elif( engine in [ 'demo-regex' , 'smoking-regex' ] ):
        nlp_pipeline = spacy.blank("en")
        med_pyrush = nlp_pipeline.add_pipe( "medspacy_pyrush" )
        target_matcher = nlp_pipeline.add_pipe( "medspacy_target_matcher" )
        ## ####
        ## Engine-specific rules
        if( engine in [ 'demo-regex' ] ):
            target_rules = [
                TargetRule(literal="abdominal pain", category="PROBLEM"),
                TargetRule("stroke", "PROBLEM"),
                TargetRule("hemicolectomy", "TREATMENT"),
                TargetRule("Hydrochlorothiazide", "TREATMENT"),
                TargetRule("colon cancer", "PROBLEM"),
                TargetRule("metastasis", "PROBLEM"),
                TargetRule("diabetes", "PROBLEM",
                           pattern = r"type (i|ii|1|2|one|two) (dm|diabetes mellitus)")
            ]
        elif( engine in [ 'smoking-regex' ] ):
            target_rules = [
                # Smoking terminology (current/active smoking)
                TargetRule("smoking", "SMOKER",
                          pattern=r"\b(smokes?|smoking|smokers?|smoked)\b"),
                
                # Tobacco products (indicates smoking behavior)
                TargetRule("tobacco products", "SMOKER",
                          pattern=r"\b(tobaccos?|cigarettes?|cigs?|pipes?|cigars?|nicotine|tob)\b"),
                
                # Negation of smoking
                TargetRule("negated smoking", "NON_SMOKER",
                          pattern=r"\b(no|non|not|never|negative)\W*(smoker|smoking|smoked|tobacco)\b"),
                
                # Compound non-smoker term
                TargetRule("nonsmoker", "NON_SMOKER",
                          pattern=r"\bnonsmoker\b"),
                
                # Clinical denial documentation
                TargetRule("denies smoking", "NON_SMOKER",
                          pattern=r"\bdenies\W*(smoking|tobacco)\b"),
                
                # Postfix negation (substance + negation)
                TargetRule("substance negation", "NON_SMOKER",
                          pattern=r"\b(tobacco|smoke|smoking|nicotine)\W*(never|no)\b"),
                
                # Negative contraction
                TargetRule("doesn't smoke", "NON_SMOKER",
                          pattern=r"\bdoes(n't| not) smoke\b"),
                
                # Quantitative negation (zero pack years)
                TargetRule("zero smoking", "NON_SMOKER",
                          pattern=r"\b(0|zero)\W*smokers?\b"),
            ]
        ## ####
        ## 
        target_matcher.add( target_rules )
    else:
        ## If the engine name is malformed, then load a default
        nlp_pipeline = medspacy.load( 'en_core_web_sm' , enable = [ 'tagger' ,
                                                                    'lemmatizer' ] )
        ##umls_dir = medspacy.util.get_quickumls_demo_dir('en')
        ##umls_dir = 'umls-ctakes'
        umls_dir = 'umls-quick-decovri'
        nlp_pipeline.add_pipe( 'medspacy_quickumls' ,
                               before = 'medspacy_context' ,
                               config = { "best_match" : True ,
                                          "result_type": "group" ,
                                          "quickumls_fp" : umls_dir } )
    print( '{}'.format( nlp_pipeline.pipe_names ) )
    ##sys.exit( 1 )
    ############################
    ## Create a type system
    ## - https://github.com/dkpro/dkpro-cassis/blob/master/cassis/typesystem.py
    ############
    ## ... for tokens
    if( typesDir is None ):
        typesystem = cassis.TypeSystem()
    else:
        with open( os.path.join( args.typesDir , 'Sentence.xml' ) , 'rb' ) as fp:
            typesystem = cassis.load_typesystem( fp )
    ########
    TokenAnnotation = typesystem.create_type( name = 'uima.tt.TokenAnnotation' , 
                                              supertypeName = 'uima.tcas.Annotation' )
    typesystem.add_feature( type_ = TokenAnnotation ,
                            name = 'text' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = TokenAnnotation ,
                            name = 'lemma' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = TokenAnnotation ,
                            name = 'partOfSpeech' , 
                            rangeTypeName = 'uima.cas.String' )
    ########
    EntityMention = typesystem.create_type( name = 'uima.tt.EntityMention' , 
                                              supertypeName = 'uima.tcas.Annotation' )
    typesystem.add_feature( type_ = EntityMention ,
                            name = 'text' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = EntityMention ,
                            name = 'entityType' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = EntityMention ,
                            name = 'ontologyConcept' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = EntityMention ,
                            name = 'confidence' , 
                            rangeTypeName = 'uima.cas.Double' )
    typesystem.add_feature( type_ = EntityMention ,
                            name = 'polarity' , 
                            rangeTypeName = 'uima.cas.Integer' )
    ############################
    ## Iterate over the files, covert to CAS, and write the XMI to disk
    """ 
    filenames = [ os.path.basename( f ) for f in glob.glob( os.path.join( inputDir ,
                                                                          '*.txt' ) ) ]
    for filename in tqdm( filenames ):
        if( verboseFlag ):
            print( '\n==== File:  {} ===='.format( filename ) )
        xmi_filename = re.sub( '.txt$' ,
                               '.xmi' ,
                               filename )
        with open( os.path.join( inputDir , filename ) , 'r' ) as fp:
            note_contents = fp.read()
        """

    csv_files = [f for f in glob.glob(os.path.join(inputDir, '*.csv'))]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Identify the text column and patient ID column
        text_col = None
        id_col = None
        
        for candidate in ['note_text', 'text', 'document', 'notes']:
            if candidate in df.columns:
                text_col = candidate
                break
        for candidate in ['patient_id', 'subject_id', 'id', 'patient']:
            if candidate in df.columns:
                id_col = candidate
                break

        if text_col is None or id_col is None:
            print(f"Could not find both text and patient ID columns in {csv_file}, skipping.")
            continue

        # Group all notes for each patient into one string
        grouped = df.groupby(id_col)[text_col].apply(lambda x: "\n\n".join(map(str, x))).reset_index()

        for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
            patient_id = str(row[id_col])
            note_contents = str(row[text_col])
            if not note_contents.strip():
                continue

            xmi_filename = f"{os.path.basename(csv_file).replace('.csv','')}_{patient_id}.xmi"

            if verboseFlag:
                print(f"\n==== Processing patient {patient_id} from {csv_file} ====")

            cas = cassis.Cas(typesystem=typesystem)
            cas.sofa_string = note_contents
            cas.sofa_mime = "text/plain"

            regex_note = nlp_pipeline(note_contents)

            # Add entity annotations
            for ent in regex_note.ents:
                if verboseFlag:
                    print(f"\t{ent.text}\t{ent.label_}\t{getattr(ent._, 'target_rule', None)}")
                cas.add_annotation(EntityMention(
                    begin=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    entityType=ent.label_,
                    polarity=1
                ))

            # Write one XMI per patient
            cas.to_xmi(path=os.path.join(outputDir, xmi_filename), pretty_print=True)

        ########################
        ## Tokens
        ## - https://spacy.io/api/token
        for token in regex_note:
            ## ####
            ## We probably don't need tokens to show up in our XMI
            ## output.  If we change our mind about that, we can
            ## always flip this boolean.
            add_tokens_to_xmi = False
            ##
            if( add_tokens_to_xmi ):
                cas.add_annotation( TokenAnnotation( begin = token.idx , 
                                                     end = token.idx + token.__len__() ,
                                                     text = token.text ,
                                                     lemma = token.lemma_ ,
                                                     partOfSpeech = token.pos_ ) )
        ########################
        ## Concepts of Interest
        if( verboseFlag ):
            print( '\n---- Entities ----\n' )
        for ent in regex_note.ents:
            if( verboseFlag ):
                print( '\t{}\t{}\t{}'.format( ent.text ,
                                              ent.label_ ,
                                              ent._.target_rule ) )
            cas.add_annotation( EntityMention( begin = ent.start_char , 
                                               end = ent.end_char ,
                                               text = ent.text ,
                                               entityType = ent.label_ ,
                                               polarity = 1 ) )
        cas.to_xmi( path = os.path.join( outputDir , xmi_filename ) ,
                    pretty_print = True )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description = 'Simple medspaCy pipeline for doing rudimentary NLP' )
    parser.add_argument( '-v' , '--verbose' ,
                         help = "print more information" ,
                         action = "store_true" )
    parser.add_argument( '-t' , '--types-dir' , default = None ,
                         dest = 'typesDir' ,
                         help = 'Directory containing the systems files need to be loaded' )
    parser.add_argument( '-i' , '--input-dir' , required = True ,
                         dest = 'inputDir' ,
                         help = 'Input directory containing plain text files to sectionize' )
    parser.add_argument( '-o' , '--output-dir' , required = True ,
                         dest = 'outputDir',
                         help = 'Output directory for writing CAS XMI files to' )
    parser.add_argument( '-p' , '--pipeline-engine' ,
                         default = 'default' ,
                         dest = 'engine',
                         help = 'The trained pipeline engine to process these notes through' )
    args = parser.parse_args()
    if( not os.path.exists( args.outputDir ) ):
        try:
            os.makedirs( args.outputDir )
        except OSError as e:
            log.error( 'OSError caught while trying to create output folder:  {}'.format( e ) )
        except IOError as e:
            log.error( 'IOError caught while trying to create output folder:  {}'.format( e ) )
    main( os.path.abspath( args.inputDir ) ,
          os.path.abspath( args.outputDir ) ,
          None if( args.typesDir == None ) else os.path.abspath( args.typesDir ) ,
          engine = args.engine ,
          verboseFlag = args.verbose )

# python3 medspaCy_regex.py -i /scratch/jkbrook/NLP/Mod_4_ML/input -o /scratch/jkbrook/NLP/Mod_4_ML/output -p smoking-regex -v
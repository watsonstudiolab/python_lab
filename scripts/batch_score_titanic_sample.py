#!/usr/bin/python

import sys, os, pickle
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
import dsx_core_utils, re, jaydebeapi
from sqlalchemy import *
from sqlalchemy.types import String, Boolean

# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

def batch_score(args):

  # define variables
  # args = {'execution_type': 'DSX', 'target': '/datasets/test-results-2.csv', 'source': '/datasets/test_df_unlabeled.csv', 'output_type': 'Localfile', 'output_datasource_type': '', 'sysparm': '', 'remoteHost': '', 'remoteHostImage': '', 'livyVersion': 'livyspark2'}
  input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("source")))
  output_data = os.getenv("DEF_DSX_DATASOURCE_OUTPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("target")))
  model_path = os.getenv("DSX_PROJECT_DIR") + os.path.join("/models", os.getenv("DSX_MODEL_NAME","Titanic_Prediction_Model"), os.getenv("DSX_MODEL_VERSION","4"),"model")

  # load the input data
  dataframe = pd.read_csv(input_data)
  
  # load the model from disk
  loaded_model = joblib.load(open(model_path, 'rb'))
  
  # predictions
  scoring_result = loaded_model.predict(dataframe)
  
  # save scoring result to given target
  scoring_df = pd.DataFrame(scoring_result)
  
  # save output to csv
  scoring_df.to_csv(output_data, encoding='utf-8')

  return scoring_df.to_json()
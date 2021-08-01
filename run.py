# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import click
import argparse

PROJECTS_LIST = './projects.json'

parser = argparse.ArgumentParser()
parser.add_argument('--type', help='Specify either "similarity" or "different" to get that type of recommendations')
parser.add_argument('--project_id', help='Project ID of the project to get recommendations of')
args = parser.parse_args()

def setup():
  # Importing Data
  df = pd.read_json(PROJECTS_LIST)

  # Changing null tags to empty lists
  df.loc[df['tags'].isnull(),['tags']] = df.loc[df['tags'].isnull(),'tags'].apply(lambda x: [])

  # Changing null themes to empty lists
  df.loc[df['themes'].isnull(),['themes']] = df.loc[df['themes'].isnull(),'themes'].apply(lambda x: [])

  # Merging themes and tags
  df['themetag'] = df['themes'] + df['tags']

  # Changing themetag to a string 
  df['themetag'] = df['themetag'].agg(lambda x: ';'.join(map(str, x)))

  # create binary indicators for each theme/tag
  # source: https://datascience.stackexchange.com/questions/14847/multiple-categorical-values-for-a-single-feature-how-to-convert-them-to-binary-u
  df_stack = df[df['themetag'] != '(no themetag listed)'].set_index('portfolioId').themetag.str.split(';', expand = True).stack()
  df_explode = pd.get_dummies(df_stack, prefix = 'g').groupby(level = 0).sum().reset_index()
  del df_stack

  # vectors of theme/tags
  df_explode['themetag_vector'] = df_explode.iloc[:,1:].values.tolist()

  # Adding list of vectors to data 
  df = df.merge(df_explode[['portfolioId','themetag_vector']], on = 'portfolioId', how = 'left')

  # Deleting rows with no themes or tags
  df = df[df.themetag != ";"]

  # Converting theme/tags from string to list
  df['themetaglist'] = df.themetag.map(lambda x: x.split(';'))

  return df

# compute Jaccard Index to get 5 most similar projects 
def get_similar_projects(target_project,df):
  target_themetag_list = df[df.portfolioId == target_project].themetaglist.values[0]
  themetag_list_sim = df[['portfolioId','showcaseName','themetaglist','themetag']]
  themetag_list_sim['jaccard_sim'] = themetag_list_sim.themetaglist.map(lambda x: len(set(x).intersection(set(target_themetag_list))) / len(set(x).union(set(target_themetag_list))))
  print(f'Projects most similar to {target_project} based on themetag:')
  text = ','.join(themetag_list_sim.sort_values(by = 'jaccard_sim', ascending = False).head(25)['themetag'].values)
  recommended=themetag_list_sim.sort_values(by = 'jaccard_sim', ascending = False).head(6)
  recommended=recommended[1:]
  return recommended['portfolioId']

# compute Jaccard Index to get least similar project (Inspire Me)
def get_diff_projects(target_project,df):
  target_themetag_list = df[df.portfolioId == target_project].themetaglist.values[0]
  themetag_list_sim = df[['portfolioId','showcaseName','themetaglist','themetag']]
  themetag_list_sim['jaccard_sim'] = themetag_list_sim.themetaglist.map(lambda x: len(set(x).intersection(set(target_themetag_list))) / len(set(x).union(set(target_themetag_list))))
  print(f'Projects least similar to {target_project} based on themetag:')
  text = ','.join(themetag_list_sim.sort_values(by = 'jaccard_sim', ascending = False).head(25)['themetag'].values)
  recommended=themetag_list_sim.sort_values(by = 'jaccard_sim', ascending = True).head(1)
  return recommended['portfolioId']

def main(args):
    df = setup()
    if int(args.project_id) in df.values:
      if args.type == "similarity":
        results = get_similar_projects(int(args.project_id), df)
      else:
        results = get_diff_projects(int(args.project_id), df)
      print(results)
    else:
      print("Please input a valid PortfolioID or a ProjectID")
    
if __name__ == "__main__":
    main(args)

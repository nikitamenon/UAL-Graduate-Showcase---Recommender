# UAL-Graduate-Showcase---Recommender (CLI-Driven)
A recommender that allows viewers to get recommendations for similar projects based on the the projects they are viewing on the UAL Graduate showcase. This similarity is determined by the tags and themes assigned to each project. 
# Motivation
The project seeks to help viewers see a variety of projects that might interest them, and also helps students gain more visibility since the recommender pushes projects to the viewer based on similarity and without bias. 
# Language Used
Python 3.7
# Utilities Used: 
- Pandas
- Numpy
- Warnings
- Doc2Vec, TaggedDocument from Gensim.models.doc2vec
- Word_tokenize from Nltk.tokenize
- Stopwords from Nltk.corpus
# Installation
pip install -r requirements.txt
# How to Use
Similarity: python run.py --type="similarity" --project_id=VALID PROJECT ID
  
Different: python run.py --type="different" --project_id=VALID PROJECT ID
  
  
The three key variables used from the dataframe are portfolioId, Tags (list) and Themes (list). For this project, the dataset imported needs to be a .json file. The code first replaces all "None" values in the tags/themes columns with empty lists and then tags and themes are combined into a single column called themetag. Each theme/tag is assigned vector values and these vectors are added back into the dataset. Any project with no themes or tags will be dropped since the model is unable to recommend projects without these variables. Based on the similarity score of (Jaccard Index), the top 5 highest scoring projects are recommended for each project. The model outputs one project of the lowest Jaccard score which will be the most dissimilar project from what the viewer is presently viewing and this would be an "inspire me" recommendation. 
# Help
For any further clarifications, contact Nikita at nikita21menon@gmail.com

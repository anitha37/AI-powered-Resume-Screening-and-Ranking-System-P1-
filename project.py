import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string

# Load the spacy English model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text (removing stopwords, punctuation, and lemmatizing)
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase and parse it
    processed_text = " ".join([token.lemma_ for token in doc if token.text not in stopwords.words('english') and token.text not in string.punctuation])
    return processed_text

# Load dataset (Resumes and Job Descriptions)
# Sample data: Replace this with loading your actual dataset
resumes_data = [
    {"resume_text": "Experienced software engineer with skills in Python, Java, and AI."},
    {"resume_text": "Data scientist skilled in machine learning, Python, and deep learning."},
    {"resume_text": "Junior software developer with knowledge in Java and web development."},
    {"resume_text": "Experienced machine learning engineer, specializing in AI models."}
]

jobs_data = [
    {"job_description": "Looking for a senior software engineer with experience in Python and AI."},
    {"job_description": "We need a data scientist proficient in machine learning and deep learning."}
]

# Convert data to pandas DataFrame
resumes_df = pd.DataFrame(resumes_data)
jobs_df = pd.DataFrame(jobs_data)

# Preprocess resumes and job descriptions
resumes_df['processed_resume'] = resumes_df['resume_text'].apply(preprocess_text)
jobs_df['processed_job'] = jobs_df['job_description'].apply(preprocess_text)

# Combine resumes and job descriptions into one corpus for vectorization
corpus = resumes_df['processed_resume'].tolist() + jobs_df['processed_job'].tolist()

# Vectorize the text using TF-IDF (Term Frequency - Inverse Document Frequency)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Split the TF-IDF matrix into separate matrices for resumes and jobs
resume_tfidf = tfidf_matrix[:len(resumes_df)]
job_tfidf = tfidf_matrix[len(resumes_df):]

# Function to rank resumes for each job description
def rank_resumes(job_index):
    job_vector = job_tfidf[job_index]  # Get the vector for the job description
    similarities = cosine_similarity(job_vector, resume_tfidf)  # Calculate cosine similarity
    ranked_resumes = similarities.flatten().argsort()[::-1]  # Sort resumes based on similarity
    return ranked_resumes

# Rank resumes for each job description
for job_index in range(len(jobs_df)):
    print(f"\nRanking for Job '{jobs_df['job_description'][job_index]}':")
    ranked_resumes = rank_resumes(job_index)
    
    # Output the top 5 ranked resumes (if available)
    for idx in ranked_resumes[:5]:
        print(f"Resume {idx+1}: {resumes_df['resume_text'][idx]}")


# import pandas as pd
# from transformers import pipeline

# # Load the uploaded CSV file
# file_path = "./model_inference_mult.csv"
# data = pd.read_csv(file_path)

# # Initialize the sentiment analysis pipeline
# distilled_student_sentiment_classifier = pipeline(
#     model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
#     top_k=None
# )

# data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# # Define a function to get sentiment score and map to 0-1 range
# def get_scaled_sentiment_score(sentence):
#     # Get the model output
#     result = distilled_student_sentiment_classifier(sentence)[0]
    
#     # Define label-to-score mapping
#     label_score_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    
#     # Calculate the original weighted sentiment score
#     sentiment_score = sum(label_score_mapping[item['label']] * item['score'] for item in result)
    
#     # Scale it to [0, 1] range
    
#     return sentiment_score

# # Apply the sentiment score calculation for each sentence
# data['Sentiment_Score'] = data['Sentence'].apply(get_scaled_sentiment_score)

# # Save the updated dataframe to a new CSV
# output_path = "./model_inference_mult_with_sentiment_score.csv"
# data.to_csv(output_path, index=False)

# print("Sentiment scores have been added and saved to:", output_path)

import pandas as pd
import torch  # To check for CUDA availability
from transformers import pipeline

# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU

# Load the uploaded CSV file
file_path = "./model_inference_annotated.csv"
data = pd.read_csv(file_path)

# Initialize the sentiment analysis pipeline with CUDA support if available
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    device=device,  # Set the device to GPU if available, otherwise CPU
    top_k=None
)

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Define a function to get sentiment score and map to 0-1 range
def get_scaled_sentiment_score(sentence):
    # Get the model output
    result = distilled_student_sentiment_classifier(sentence)[0]
    
    # Define label-to-score mapping
    label_score_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    
    # Calculate the original weighted sentiment score
    sentiment_score = sum(label_score_mapping[item['label']] * item['score'] for item in result)
    
    return sentiment_score

# Apply the sentiment score calculation for each sentence
data['Sentiment_Score'] = data['Sentence'].apply(get_scaled_sentiment_score)

# Save the updated dataframe to a new CSV
output_path = "./model_inference_annotated_with_sentiment_score.csv"
data.to_csv(output_path, index=False)

print("Sentiment scores have been added and saved to:", output_path)


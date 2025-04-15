# Install necessary libraries
!pip install -q transformers sentencepiece

# Import required modules
from transformers import pipeline
import textwrap

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize news article
def summarize_article(article_text, max_length=130, min_length=30):
    """
    Summarizes a news article using a pretrained transformer model.

    Parameters:
        article_text (str): The full news article text.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    if len(article_text.strip()) == 0:
        return "Error: Article text is empty."

    summary = summarizer(article_text, max_length=max_length, min_length=min_length, do_sample=False)
    return textwrap.fill(summary[0]['summary_text'], width=100)

# Example usage
article = """
Former President Barack Obama spoke at the climate summit today, emphasizing the urgent need for global cooperation
to combat climate change. He acknowledged progress made since the Paris Agreement but urged nations to move faster.
Obama also highlighted the role of young people and innovation in driving environmental change. His speech received
a standing ovation from the audience.
"""

print("Original Article:\n")
print(textwrap.fill(article, width=100))

print("\nSummarized Article:\n")
print(summarize_article(article))

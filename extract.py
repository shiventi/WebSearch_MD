import json

with open('web_questions.json', 'r') as file:
    data = json.load(file)

# Extract text within quotes from the "description" field
questions_and_answers = []
for entry in data:
    target_value = entry.get('targetValue')
    utterance = entry.get('utterance')
    if target_value and utterance:
        descriptions = [item.strip('() \n"list') for item in target_value.split('(description') if item.strip()]
        # Join multiple descriptions into one line
        answer = ' '.join(descriptions)
        # Format the output as "question: answer"
        question_and_answer = f'{utterance}:{answer}'
        questions_and_answers.append(question_and_answer)

# Write each extracted question and answer on a new line in a file
with open("answers.txt", "w") as output_file:
    for question_and_answer in questions_and_answers:
        output_file.write(question_and_answer)
        output_file.write("\n")




####################



import spacy
from bs4 import BeautifulSoup
import requests
import re
from better_profanity import profanity
import time
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pymongo
import torch

def extract_all_span_text(input_text):
    soup = BeautifulSoup(input_text, 'html.parser')
    
    spans = soup.find_all('span')
    span_texts = [span.get_text() for span in spans]
    
    return ''.join(span_texts)

def extract_innermost_span_text(input_text):
    soup = BeautifulSoup(input_text, 'html.parser')
    
    innermost_span = soup.find_all('span')[1]
    innermost_text = innermost_span.get_text()
    
    return innermost_text

def generate_summary(paragraph, model_name="Falconsai/text_summarization"):
    summarization_pipe = pipeline("summarization", model=model_name)

    summary = summarization_pipe(paragraph, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summarized_text = summary[0]['summary_text']

    return summarized_text
####################

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")


def link_finder(query):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
        }

        url = f'https://www.google.com/search?q={query}&ie=utf-8&oe=utf-8&num=10'
        html = requests.get(url, headers=headers)
        html.raise_for_status()  # Raise an HTTPError for bad responses

        soup = BeautifulSoup(html.text, 'html.parser')
        all_data = soup.find_all("div", {"class": "g"})

        result_data = []
        seen_links = set()
        position = 0

        def extract_data(data):
            nonlocal position
            try:
                link = data.find('a').get('href')
            except AttributeError as e:
                print(f"Error extracting link: {e}")
                return

            if link and link.find('https') != -1 and link.find('http') == 0 and link.find('aclk') == -1:
                if link not in seen_links:
                    position += 1
                    result_entry = {
                        "link": link,
                        "position": position,
                        "title": None,
                        "description": None
                    }

                    try:
                        result_entry["title"] = data.find('h3', {"class": "DKV0Md"}).text
                    except AttributeError as e:
                        print(f"Error extracting title: {e}")

                    try:
                        result_entry["description"] = data.find("div", {"class": "lyLwlc"}).text
                    except AttributeError as e:
                        print(f"Error extracting description: {e}")

                    result_data.append(result_entry)
                    seen_links.add(link)

        for data in all_data:
            extract_data(data)

        return [item["link"] for item in result_data][:3]

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def page_content(link):
    webpage = requests.get(link)
    soup = BeautifulSoup(webpage.content, 'html.parser')
    
    all_texts = [element.get_text() for element in soup.find_all(text=True)]

    return all_texts

def question_answer(question, linked_links):
    qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
    
    highest_score = 0
    best_answer = None
    
    for link in linked_links:
        try:
            pc = page_content(link)
            pc = " ".join(pc)
            if pc == None or pc == "":
                continue
            else:
                result = qa_model(question=question, context=str(pc))
                score = float(result["score"])
                
                if score > 1:
                    pass
                
                if score > highest_score:
                    highest_score = score
                    best_answer = result
        except Exception as e:
            return (f"Error processing link {link}: {e}")
    
    return best_answer

def question_answer_c(question, context):
    qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
    
    result = qa_model(question=question, context=str(context))
    score = float(result["score"])

    return result, score


def web_search(url, query, element_type=None, class_name=None):
    full_url = f"{url}?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(full_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    search_results = soup.find_all(element_type, class_=class_name)
    
    cleaned_results = []

    for result in search_results:
        b_tag = result.find('b')
        if b_tag:
            return b_tag.get_text()
        else:
            cleaned_results.append(result.get_text())
    
    return search_results


def answer(user_input):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open("questions.txt", "r", encoding="utf-8") as file:
        all_sentences = [line.strip() for line in file.readlines()]

    all_sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(user_input_embedding, all_sentence_embeddings)[0].tolist()
    max_similarity_index = similarities.index(max(similarities))
    most_similar_sentence = all_sentences[max_similarity_index]
    max_similarity = similarities[max_similarity_index]

    with open("answers.txt", "r") as a:
        b = a.readlines()

    if float(max_similarity) <= float(0.81):
        a_result = link_finder(user_input)
        d_result = web_search("https://www.google.com/search", user_input, "span", "hgKElc")
        z_result = web_search("https://www.google.com/search", user_input, "div", "WFxqwc BGdUVb OTFaAf")

        try:
            y = str(z_result[0])
            x = extract_all_span_text(y)
            w = question_answer_c(user_input, x)

            if not d_result:
                if not x:
                    e = question_answer(user_input, a_result)
                    if e and "answer" in e:
                        f = e["answer"].replace(".", "").replace(",", "").replace("!", "")
                        return f
                    else:
                        return "NOTHING"
                else:
                    an = w[0]["answer"]
                    if "wikipedia" in an.lower():
                        ab = extract_innermost_span_text(y)
                        return ab
                    else:
                        return an
            else:
                return d_result

        except Exception as ex:
            if d_result:
                return d_result
            else:
                e = question_answer(user_input, a_result)
                if e and "answer" in e:
                    f = e["answer"].replace(".", "").replace(",", "").replace("!", "")
                    return f
                else:
                    return "NOTHING"

    else:
        for i in b:
            if most_similar_sentence in i:
                c = i.split(": ")
                return c[1]




def find_most_similar_sentence(user_input_sentence, all_sentences):
    user_input_embedding = model.encode(user_input_sentence, convert_to_tensor=True)

    all_sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(user_input_embedding, all_sentence_embeddings)[0]

    max_similarity_index = torch.argmax(similarities).item()

    most_similar_sentence = all_sentences[max_similarity_index]
    max_similarity = similarities[max_similarity_index].item()

    return most_similar_sentence, max_similarity

# Set Streamlit page configuration outside of the main function
st.set_page_config(
    page_title="Question Answering",
    page_icon="🦁",
)

@st.cache_resource
def init_connection():
    # Establish MongoDB connection using credentials from Streamlit secrets
    return pymongo.MongoClient(st.secrets["mongodb"]["user"])

# Initialize the MongoDB client
client = init_connection()

@st.cache_data(ttl=600)
def insert_in_db(user_input):
    try:
        # Load all sentences from the questions.txt file
        with open("questions.txt", "r", encoding="utf-8") as file:
            all_sentences = [line.strip() for line in file.readlines()]

        # Access the question collection in the MongoDB database
        db = client.questiondb
        question = db.question

        # Find the most similar sentence
        b = find_most_similar_sentence(str(user_input), all_sentences)

        # Check similarity threshold and insert into the database
        if float(b[1]) > 0.6:
            return "ALL GOOD"
        else:
            new_question = {"question": user_input}
            question.insert_one(new_question)
            return "ADDED"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Rest of your main function remains the same
    st.title("Question Answering")

    # Display a warning about the app's potential slowness
    st.warning("This app may run slowly as it fetches data from the web", icon="⚠️")

    # Get user input
    user_input = st.text_input("Enter your sentence:")

    # Check for profanity in user input
    s = profanity.contains_profanity(str(user_input))

    if not s:
        # If no profanity, proceed with finding the answer and inserting into the database
        if st.button("Get Answer"):
            with st.spinner("Finding answer..."):
                answer_result = answer(user_input)
                time.sleep(2)
            st.success("Answer found!")
            st.write("Answer:", answer_result)

            # Insert the user input into the database
            result_message = insert_in_db(user_input)
            st.write(result_message)
    else:
        # Display an error message for profanity
        st.error("NO BAD WORDS ALLOWED!", icon='🚨')

if __name__ == "__main__":
    main()

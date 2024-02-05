from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def find_most_similar_sentence(user_input_sentence, all_sentences):
    user_input_embedding = model.encode(user_input_sentence, convert_to_tensor=True)

    all_sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(user_input_embedding, all_sentence_embeddings)[0]

    max_similarity_index = torch.argmax(similarities).item()

    most_similar_sentence = all_sentences[max_similarity_index]
    max_similarity = similarities[max_similarity_index].item()

    return most_similar_sentence, max_similarity

with open("questions.txt", "r", encoding="utf-8") as file:
    all_sentences = [line.strip() for line in file.readlines()]

#print(all_sentences)




client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000")
db = client.questiondb

question = db.question

while True:
    a = input("QUESTION: ")
    b = find_most_similar_sentence(str(a), all_sentences)
    if float(b[1]) > 0.6:
        print("ALL GOOD!")
    else:
        new_question = {"question": a}

        question.insert_one(new_question)

        #cursor = question.find()

        #for document in cursor:
            #print(document["question"])
        print("QUESTION ADDED")



#pip install pip install sentence-transformers
#pip install numpy

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Path to files to search from (must be .json format)
path = ""

filtered_json = []
with open(path, 'r') as file:
    for line in file:
#         if '"Complete Specification"' in line:
            filtered_json.append(line)


#Enter the query here
query = r""
docs = filtered_json

#Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

#Encode query
query_emb = model.encode(query)

# Run these commands only for the first time to encode the files to seach from

#doc_emb = model.encode(docs)
#doc_emb = model.encode(filtered_json)
#print("After corpus_embedding")
#filtered_json = np.load("filtered_json.npy")
#all_embeddings = doc_emb
#all_embeddings = np.array(all_embeddings)
#np.save('embeddings(model2, test_wala).npy', all_embeddings)

#Load the numpy file that will be saved only one time by the commands run the first time
doc_emb = np.load(r"")

#Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#print(len(scores))

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)


'''
# LINE-WISE EMBEDDING

for i in range(5):
    result = list()

    result.append(doc_score_pairs[i][0])

    line_result = str(result[0]).split(".")
#    print(line_result[6])

    line_emb = model.encode(line_result)

    line_score = util.dot_score(query_emb, line_emb)[0].cpu().tolist()


    line_score_pairs = list(zip(line_result, line_score))

    #Sort by decreasing score
    line_score_pairs = sorted(line_score_pairs, key=lambda x: x[1], reverse=True)

    print(doc_score_pairs[i])
    for i in range(5):
        print(f"\nMOST RELEVANT LINES : {line_score_pairs[i]}")
'''
        

# LINE-PARAGRAPH CHECK

query_list = query.split(".")
line_fullstop_list = list()

for i in range(len(query_list)):
    query_line_emb = model.encode(query_list[i])
    scores_line_emb = util.dot_score(query_line_emb, doc_emb)[0].cpu().tolist()
    line_doc_score_pairs = list(zip(docs, scores_line_emb))
    line_fullstop_list.append(line_doc_score_pairs)

sum = 0
average = list()

for j in range(len(docs)):
    for i in range(len(query_list)):
            sum = (sum + line_fullstop_list[i][j][1])
    average.append(sum/len(query_list))

line_doc_score_pairs = list(zip(docs, average))
line_doc_score_pairs = sorted(line_doc_score_pairs, key=lambda x: x[1], reverse=True)

#for i in range(5):
#     print(line_doc_score_pairs[i])
print(len(query_list))


#line_doc_score_pairs = list(zip(query_list, docs, scores_line_emb))
##Sort by decreasing score
#line_doc_score_pairs = sorted(line_doc_score_pairs, key=lambda x: x[1], reverse=True)
##result.append(line_doc_score_pairs[0])
    
#print(len(line_doc_score_pairs))




#Output passages & scores
#for doc, score in doc_score_pairs:
#    print(score, doc)

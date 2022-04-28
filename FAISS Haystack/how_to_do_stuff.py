import faiss
import numpy as np
import pandas as pd
from haystack.document_stores import FAISSDocumentStore

"""
FYI File - How to use FAISS and Haystack

There is no easy way to get an existing FAISS index into Haystack, so instead 
I will create the index with Haystack and get what I need out of it for 
de-duplicating docs via similarity and keyword semantic search.  (The goal is 
to execute a keyword semantic search, but it would be more efficient to first 
de-duplicate by similarity so that fewer docs need to be searched.)

To start off, here is an example of how to do things with raw FAISS...
"""

# the text column from here is required by Haystack for storage in the DB
# the search_hit_id column from here can be used as a custom ID in FAISS
df = pd.read_csv("temp_duplicates.csv")
df.dropna(axis=0, inplace=True)

# assume you already have embeddings saved as a NP array of shape (nbr_embeddings, dim)
embed = np.load('embeddings.npy').astype(np.float32)
print(embed.shape)

d = embed.shape[1]  # dimensionality of the embeddings
nlist = 3  # how many Voroni cells, essentially the k in k-means
# X * sqrt(nbr_embeddings) is a good starting place for nlist, where X = 4 or some other constant
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)  # faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)

# change how many neighboring Voroni cells to search
index.nprobe = 2

# NOTE - if using with Haystack, must use Haystack to train the index, cannot do it here
# train index first...
index.train(embed)

"""
Stop! 

Ordinarily, you would proceed by adding the embeddings.  Bu Haystack doesn't like that - 
it requires you add them using its write_documents function.  But that's ok.  At the bottom of 
this script, I will prove that the result is identical to what you would get if you were 
to do it here.  If you want to try the code below, uncomment it and comment out the Haystack 
stuff.
"""

'''
# ...then add embeddings to index
index.add(embed)
assert(index.ntotal == embed.shape[0])

# this line is required to map from index back to embedding
index.make_direct_map()
# and here is how you can reconstruct the embedding from the index
index.reconstruct(0)[:]

# save the index
faiss.write_index(index, 'doc_store_index.faiss')

# Now you can do a knn search
k = 5
xq = embed[0, :]  # pick some embedding to serve as a query
# the query must be an array
if len(xq.shape) < 2:
    xq = np.array([xq])
distances, indices = index.search(xq, k)  # search
print(indices)
# Or a range search
dist = 60.
limits, distances, indices = index.range_search(xq, dist)  # search
print(distances)
print(indices)

# if you want to compress the vector storage space, instead of using Flat index, use Product Quantization (PQ)
# IVF reduces the scope of the search, while PQ approximates the distance calculation instead of using L2
# PQ does this by slicing the vector, clustering the slices, and then returning the centroid ID vector
m = 8  # number of centroid IDs in final compressed vectors
bits = 8  # number of bits in each centroid
pq_quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
pq_index = faiss.IndexIVFPQ(pq_quantizer, d, nlist, m, bits)
pq_index.nprobe = 5
# train the index first...
pq_index.train(embed)
# ...then add the embeddings
pq_index.add(embed)

D, I = pq_index.search(xq, k)
print(I)
dist = 60.
limits, distances, indices = pq_index.range_search(xq, dist)  # search
print(distances)
print(indices)
'''

"""
Haystack does not technically support the IVFPQ type index, only the Flat, HNSW, and IVF.  
Flat is just straight pairwise L2 distance so there are no efficiencies gained.  But IVF 
uses the voroni cells, a la k-means, so that there is a speed up for larger datasets.  
However, you can replace the IndexIVFFlat with IndexIVFPQ when you create the FAISS index 
outside of Haystack and keep the faiss_index_factory_str argument = IVF{nlist},Flat and 
it will still work (or IVF{nlist},PQ).  Haystack does not seem to notice or care.  

Here I will update the FAISS index created above, but using Haystack.  Note the 
haystack_index_name.  This can be whatever you want, but it must be the same for each 
function call so Haystack knows which index you want to update.
"""

nlist = 3
d = embed.shape[1]
haystack_index_name = 'document'

document_store = FAISSDocumentStore(
    sql_url="sqlite:///faiss_document_store.db",  # defaults to SQLite, could use Postgres here instead
    embedding_dim=d,  # defaults to 768, but USE is what I used to create the embeddings so I changed it to 512
    faiss_index_factory_str=f"IVF{nlist},PQ",  # defaults to Flat, but here I'm using the inverted index, could also use HNSW
    faiss_index=index,  # existing FAISS index
    return_embedding=False,  # if true, will return normalized embedding
    index=haystack_index_name,  # name of the index in the document store (anything you want)
    similarity="dot_product",  # either dot_product (default) or cosine, Haystack normalizes both to (0,1)
    embedding_field="embedding",  # name of the embedding field (anything you want)
    progress_bar=True,
    duplicate_documents="overwrite",  # (skip, overwrite, fail) what should happen if you upload a doc whose ID is already in the index?
    faiss_index_path=None,  # where to save the index when calling .save()
    faiss_config_path=None,  # where to save the args passed here upon object initialization when .save() is called
    isolation_level=None,  # sqlalchemy parameter for create_engine()
    n_links=64,  # HNSW param only
    ef_search=20,  # HNSW param only
    ef_construction=80,  # HNSW param only
)

# you could train the index here with Haystack instead of training it above
'''
# train the index first (if it has not been trained already)...
document_store.train_index(
    documents=None,  # docs plus embeddings
    embeddings=embed,  # if you already have embeddings and don't want to pass text, put them here
    index=haystack_index_name,  # name of the index (anything you want, must match name from init)
)
'''

"""
Haystack requires the input to write_documents to be formed a specific way, with fields 'content' 
and 'embedding'.  So the first couple of lines pull in the raw text to combine it with the text 
embedding.  
"""

# haystack_input = [{'content': doc, 'embedding': embed[doc_id, :]} for doc_id, doc in enumerate(df.text.values.tolist())]
# if you have custom IDs, this is the Haystack equivalent of add_with_ids...
haystack_input = [
    {'content': row[1], 'embedding': embed[row[0], :], 'id': row[2]}
    for row in df[['text', 'search_hit_id']].itertuples()
]

# then add the documents to the index...
document_store.write_documents(
    documents=haystack_input,
    index=haystack_index_name,  # name of the index (anything you want, must match name from init)
    batch_size=10000,
    duplicate_documents='skip',  # (skip, overwrite, fail) what should happen if you upload a doc whose ID is already in the index?
)

"""
The index can be found as shown below.  You can do anything with it that you would with an ordinary 
FAISS index.
"""

assert(document_store.faiss_indexes['document'].ntotal == embed.shape[0])
# now it is safe to alter the original index, so that it can be compared to what came out of Haystack
index.add(embed)
distances, indices = index.search(np.array([embed[0, :]]), 5)
h_distances, h_indices = document_store.faiss_indexes['document'].search(np.array([embed[0,:]]), 5)
print(indices, "\n", h_indices)  # should all be equal
print(distances, "\n", h_distances)  # should all be equal

# TODO: experiment to find ideal nlist and nprobe, will trade off between accuracy and speed
# TODO: in experiment ^, can use FlatL2 for ground truth
"""
Note that if you are using custom IDs during testing, you will need to use add_with_ids 
instead of index.add().  To do this, see example here:

index = faiss.IndexFlatL2(xb.shape[1]) 
ids = np.arange(xb.shape[0])
index.add_with_ids(xb, ids)  # this will crash, because IndexFlatL2 does not support add_with_ids
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(xb, ids) # works, the vectors are stored in the underlying index

Also:  The IndexIVF sub-classes always store vector IDs. Therefore, the IndexIDMap's 
additional table is a waste of space. The IndexIVF offers add_with_ids natively.
"""


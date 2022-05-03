import logging
import os
import time
import json
import pandas as pd
import numpy as np

from exact_matches import ExactDuplicateFinder
from embeddings import USEEmbeddingModel
from faiss_index import DocumentIndexer
from near_matches import NearDuplicateFinder
from semantic_search import SimilaritySearcher
from ad_predictor import AdClassifier
from scoring import RelevancyScorer


DATA_PATH = "../Data/blackwing_3m_9k.csv"
LOG_NAME = "pipeline_logger.txt"
SYNDICATION_DISTANCE_THRESHOLD = 60.  # TODO: need to tune this
QUERY = ["electronic health record system", "epic systems"]

if os.path.exists(LOG_NAME):
    os.remove(LOG_NAME)
logging.basicConfig(
    filename=LOG_NAME,
    filemode='w',  # 'a' for append, 'w' for overwrite
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
    force=True
)
logger = logging.getLogger(LOG_NAME)


if __name__ == '__main__':
    start = time.time()

    # Step 1: reduce dataset by collapsing exact duplicates
    logger.info(f"-------- STEP 1: Find exact duplicates --------")
    logger.info(f" Reading data from {DATA_PATH}")
    edf = ExactDuplicateFinder(data_path=DATA_PATH)
    doc_id_to_hash_map, data_deduped = edf.find_exact_duplicates(
        text_col="text",
        hash_col="hash",
        doc_id_col="search_hit_id"
    )
    with open("data_doc_id_to_hash_map.json", "w") as f:
        json.dump(doc_id_to_hash_map, f)
    data_deduped.to_csv("data_deduped.csv", index=False)
    del edf
    step1_time = time.time()
    logger.info((
        f"{abs(len(doc_id_to_hash_map) - len(data_deduped))} exact duplicates found "
        f"in {round(step1_time - start, 4)} seconds."
    ))

    # Step 2: embed the documents
    logger.info(f"-------- STEP 2: Embed the documents --------")
    data_deduped = pd.read_csv("data_deduped.csv")
    model = USEEmbeddingModel(logger=logger)
    use_embeddings = model.get_embeddings(
        text_input=data_deduped['text'].tolist(),
        batch_size=256
    )
    np.save('embeddings.npy', use_embeddings, allow_pickle=True, fix_imports=False)
    del model
    step2_time = time.time()
    logger.info((
        f"{use_embeddings.shape[0]} documents embedded with {use_embeddings.shape[1]} dimensions "
        f"in {round(step2_time - step1_time, 4)} seconds."
    ))

    # Step 3: (can run in parallel with Ad Classification) create a FAISS index and document store
    logger.info(f"-------- STEP 3: Index the documents with FAISS --------")
    use_embeddings = np.load('embeddings.npy')
    data_deduped = pd.read_csv("data_deduped.csv")
    doc_indexer = DocumentIndexer(
        logger=logger,
        embeddings=use_embeddings,
        doc_texts=data_deduped.text.to_list(),
        doc_ids=data_deduped.search_hit_id.to_list()
    )
    doc_indexer.add_docs_to_index(
        nlist=3,
        nprobe=2,
        faiss_index_type='Flat',
        haystack_index_name='document',
        similarity_method='dot_product',
        duplicate_handling_method="overwrite",
        save_path=None,
        batch_size=10000,
    )
    doc_indexer.save_faiss_index(save_path='doc_store_index.faiss')
    del doc_indexer
    step3_time = time.time()
    logger.info(f"FAISS index built in {round(step3_time - step2_time, 4)} seconds.")

    # Step 4: (can run in parallel with Ad Classification) reduce the dataset by collapsing near duplicates
    logger.info(f"-------- STEP 4: Find non-exact match syndicated content --------")
    ndf = NearDuplicateFinder(data_path="data_deduped.csv", faiss_index_path="doc_store_index.faiss")
    nbr_clusters, temp = ndf.find_near_duplicates(
        distance_threshold=SYNDICATION_DISTANCE_THRESHOLD,
        cluster_col="cluster",
        cols_to_keep=["search_hit_id", "text", "cluster"]
    )
    temp.to_csv("temp_duplicates_clustered.csv", index=False)
    del ndf
    step4_time = time.time()
    logger.info((
        f"{nbr_clusters} clusters of non-exact match syndicated content found, using a distance threshold of "
        f"{SYNDICATION_DISTANCE_THRESHOLD}, in {round(step4_time - step3_time, 4)} seconds."
    ))

    # Step 5: (can run in parallel with Ad Classification) perform semantic search
    logger.info(f"-------- STEP 5: Semantic similarity search --------")
    ss = SimilaritySearcher(
        faiss_index_path="doc_store_index.faiss",
        embedding_model_path="USE_Embedding_Model",
        post_processing_scaler_path="USE_Embedding_Model/post_processing_scaler.joblib"
    )
    # TODO: if using step 4's results, replace data_deduped with that data
    ss_results = ss.text_similarity(query=QUERY, doc_ids=data_deduped.search_hit_id.to_list())
    with open("sim_results.json", "w") as f:
        json.dump(ss_results, f)
    del ss
    step5_time = time.time()
    logger.info((
        f"Similarity search for {len(QUERY)} search terms completed in {round(step5_time - step4_time, 4)} seconds."
    ))

    # Step 6: assign ad probability scores

    # Step 7: calculate final search relevancy scores
    logger.info(f"-------- STEP 7: Calculate relevancy scores --------")
    with open("sim_results.json", "r") as f:
        ss_results = json.load(f)
    with open("data_doc_id_to_hash_map.json", "r") as f:
        doc_id_to_hash_map = json.load(f)
    rs = RelevancyScorer(logger=logger, sim_search_results=ss_results, all_doc_hashes=doc_id_to_hash_map)
    results = rs.score_results()
    step7_time = time.time()
    logger.info((
        f"Relevancy scores calculated in {round(step7_time - step5_time, 4)} seconds."
    ))

    logger.info(f"Pipeline finished.  Total time elapsed: {round(time.time() - start, 4)} seconds.")

    # sense check
    for search_term, search_res in results.items():
        raw_df = pd.read_csv(DATA_PATH)
        raw_df = raw_df[['search_hit_id', 'source_name', 'headline', 'text']].copy()
        raw_df['score'] = raw_df['search_hit_id'].astype(str).map(search_res)
        raw_df.sort_values('score', ascending=False, inplace=True)
        raw_df.drop_duplicates().to_csv(f"{search_term} results.csv", index=False, float_format='%.4f')

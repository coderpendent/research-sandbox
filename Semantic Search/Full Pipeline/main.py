import logging
import os
import time
import json
import pandas as pd
import numpy as np

from data_ingestion import DataCleaner
from exact_matches import ExactDuplicateFinder
from embeddings import USEEmbeddingModel
from faiss_index import DocumentIndexer
from near_matches import NearDuplicateFinder
from semantic_search import SimilaritySearcher
from ad_predictor import AdClassifier
from scoring import RelevancyScorer


DATA_PATH = "../Data/blackwing_3m_9k.csv"
DATA_COLUMNS_TO_INGEST = ['search_hit_id', 'source_name', 'headline', 'text']
DOC_ID_COLUMN_NAME = "doc_id"
CLEAN_DATA_PATH = "../Data/blackwing_3m_9k_clean.csv"
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

    # Step 1: ingest and clean dataset
    logger.info(f"-------- STEP 1: Ingest and clean data --------")
    logger.info(f" Reading data from {DATA_PATH}")
    dc = DataCleaner(logger=logger, data_path=DATA_PATH, data_colnames=DATA_COLUMNS_TO_INGEST)
    clean_df = dc.clean_data(doc_id_colname=DOC_ID_COLUMN_NAME)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    del dc
    step1_time = time.time()
    logger.info(f"Ingested and cleaned data in {round(step1_time - start, 4)} seconds.")

    # Step 2: reduce dataset by collapsing exact duplicates
    logger.info(f"-------- STEP 2: Find exact duplicates --------")
    edf = ExactDuplicateFinder(logger=logger, data_path=CLEAN_DATA_PATH)
    doc_id_to_hash_map, data_deduped = edf.find_exact_duplicates(
        text_col="text",
        hash_col="hash",
        doc_id_col=DOC_ID_COLUMN_NAME,
        cols_to_keep=["search_hit_id"]
    )
    with open("data_doc_id_to_hash_map.json", "w") as f:
        json.dump(doc_id_to_hash_map, f)
    data_deduped.to_csv("data_deduped.csv", index=False)
    del edf
    step2_time = time.time()
    logger.info((
        f"{abs(len(doc_id_to_hash_map) - len(data_deduped))} exact duplicates found "
        f"in {round(step2_time - step1_time, 4)} seconds."
    ))
    # update for sense check
    clean_df["hash"] = clean_df[DOC_ID_COLUMN_NAME].map(doc_id_to_hash_map)
    clean_df = pd.merge(
        left=clean_df, right=data_deduped[[DOC_ID_COLUMN_NAME, "search_hit_id"]],
        how="left", on=DOC_ID_COLUMN_NAME, suffixes=["", "_y"]
    )
    clean_df["dup_removed"] = np.where(clean_df["search_hit_id_y"].isna(), 1, 0)
    clean_df.drop(clean_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

    # Step 3: embed the documents
    logger.info(f"-------- STEP 3: Embed the documents --------")
    model = USEEmbeddingModel(logger=logger)
    use_embeddings = model.get_embeddings(
        text_input=data_deduped['text'].tolist(),
        batch_size=256
    )
    np.save('embeddings.npy', use_embeddings, allow_pickle=True, fix_imports=False)
    del model
    step3_time = time.time()
    logger.info((
        f"{use_embeddings.shape[0]} documents embedded with {use_embeddings.shape[1]} dimensions "
        f"in {round(step3_time - step2_time, 4)} seconds."
    ))
    # update sense checker
    data_deduped["embedding"] = use_embeddings.tolist()
    clean_df = pd.merge(left=clean_df, right=data_deduped, how='left', on=DOC_ID_COLUMN_NAME, suffixes=["", "_y"])
    clean_df.drop(clean_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

    # Step 4: (can run in parallel with Ad Classification) create a FAISS index
    logger.info(f"-------- STEP 4: Index the documents with FAISS --------")
    doc_indexer = DocumentIndexer(
        logger=logger,
        embeddings=use_embeddings,
        doc_ids=data_deduped[DOC_ID_COLUMN_NAME].to_list()
    )
    faiss_id_to_doc_id_map = doc_indexer.add_docs_to_index(
        nlist=3, nprobe=3, faiss_index_type='Flat',
    )
    doc_indexer.save_faiss_index(save_path='doc_store_index.faiss')
    with open("faiss_id_to_doc_id_map.json", "w") as f:
        json.dump(faiss_id_to_doc_id_map, f)
    del doc_indexer
    step4_time = time.time()
    logger.info(f"FAISS index built in {round(step4_time - step3_time, 4)} seconds.")

    # Step 5: (can run in parallel with Ad Classification) reduce the dataset by collapsing near duplicates
    logger.info(f"-------- STEP 5: Find non-exact match syndicated content --------")
    ndf = NearDuplicateFinder(
        data_path="data_deduped.csv",
        faiss_id_to_doc_id_map=faiss_id_to_doc_id_map,
        faiss_index_path="doc_store_index.faiss"
    )
    nbr_clusters, data_deduped_and_clustered = ndf.find_near_duplicates(
        distance_threshold=SYNDICATION_DISTANCE_THRESHOLD,
        cluster_col="cluster",
        doc_id_col=DOC_ID_COLUMN_NAME,
        cols_to_keep=["search_hit_id", "text", "cluster"],
        reduce=False
    )
    data_deduped_and_clustered.to_csv("data_deduped_and_clustered.csv", index=False)
    del ndf
    step5_time = time.time()
    logger.info((
        f"{nbr_clusters} clusters of non-exact match syndicated content found, using a distance threshold of "
        f"{SYNDICATION_DISTANCE_THRESHOLD}, in {round(step5_time - step4_time, 4)} seconds."
    ))
    # update sense checker
    clean_df = pd.merge(
        left=clean_df, right=data_deduped_and_clustered, how='left', on=DOC_ID_COLUMN_NAME, suffixes=["", "_y"]
    )
    clean_df.drop(clean_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

    # Step 6: (can run in parallel with Ad Classification) perform semantic search
    logger.info(f"-------- STEP 6: Semantic similarity search --------")
    ss = SimilaritySearcher(
        faiss_index_path="doc_store_index.faiss",
        faiss_id_to_doc_id_map=faiss_id_to_doc_id_map,
        embedding_model_path="USE_Embedding_Model",
        post_processing_scaler_path="USE_Embedding_Model/post_processing_scaler.joblib"
    )
    ss_results = ss.text_similarity(query=QUERY)
    with open("sim_results.json", "w") as f:
        json.dump(ss_results, f)
    del ss
    step6_time = time.time()
    logger.info((
        f"Similarity search for {len(QUERY)} search terms completed in {round(step6_time - step5_time, 4)} seconds."
    ))

    # Step 7: assign ad probability scores

    # Step 8: calculate final search relevancy scores
    logger.info(f"-------- STEP 8: Calculate relevancy scores --------")
    rs = RelevancyScorer(logger=logger, sim_search_results=ss_results, all_doc_hashes=doc_id_to_hash_map)
    results = rs.score_results()
    with open("sim_results_final.json", "w") as f:
        json.dump(results, f)
    del rs
    step8_time = time.time()
    logger.info((
        f"Relevancy scores calculated in {round(step8_time - step6_time, 4)} seconds."
    ))

    logger.info(f"Pipeline finished.  Total time elapsed: {round(time.time() - start, 4)} seconds.")

    # sense check
    for search_term, search_res in results.items():
        clean_df['score'] = clean_df[DOC_ID_COLUMN_NAME].astype(int).map(search_res)
        clean_df.sort_values(['score', 'hash', 'doc_id'], ascending=False, inplace=True)
        clean_df.to_csv(f"{search_term} results.csv", index=False, float_format='%.6f')

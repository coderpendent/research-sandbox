
class RelevancyScorer:
    def __init__(self, logger, sim_search_results, all_doc_hashes):
        self.logger = logger
        self.sim_search_results = sim_search_results
        self.all_doc_hashes = all_doc_hashes

    def _convert_distance_to_similarity(self, nested_dict):
        """Inverts distance to a similarity score so that smaller distances are big and larger distances are small"""
        for k, v in nested_dict.copy().items():
            if isinstance(v, dict):  # if value is dict, go a level deeper
                nested_dict[k] = self._convert_distance_to_similarity(v)
            elif isinstance(v, float):  # only the distance scores are floats
                nested_dict[k] = 1 / v
        return nested_dict

    def _update_search_result_scores_for_missing_docs(self):
        """
        The search results do not have scores for all documents, due to filtering out exact matches.
        Therefore, the scores for the missing documents need to be obtained from the hash.  This works because the
        hash is the same for all exact matches, so the score should be the same too.  If there are still documents
        with no score after that, they were simply not returned by the similarity search and had no exact matches,
        so set their scores to 0.
        """
        # iterate over dict copy and make updates to original
        for search_term, search_res in self.sim_search_results.copy().items():

            # map hashes to scores for available docs (docs in the search results)
            new_search_res = search_res.copy()
            hash_to_score_map = {self.all_doc_hashes[k]: v for k, v in search_res.items()}

            # score the docs that are not in the search results but share a hash with a doc in the search results
            # 0 score for any docs were not in the search results and do not share a hash with a doc that was
            doc_ids_with_hash_but_no_score = set(self.all_doc_hashes.keys()) - set(search_res.keys())
            self.logger.info((
                f"{len(doc_ids_with_hash_but_no_score)} docs not in the similarity search results "
                f"but that share a hash with a doc in the similarity search results "
                f"for search term: {search_term}"
            ))
            updates_to_search_res = {
                doc_id: hash_to_score_map[self.all_doc_hashes[doc_id]]
                if self.all_doc_hashes[doc_id] in hash_to_score_map.keys() else 0
                for doc_id in doc_ids_with_hash_but_no_score
            }
            # there should be no doc IDs in the updates that were also in the search results
            assert((set(updates_to_search_res.keys()) & set(search_res.keys())) == set())
            new_search_res.update(updates_to_search_res)

            # replace the search results for this term
            self.sim_search_results[search_term] = new_search_res

    def _sort_results_by_similarity_score(self):
        """Sorts results by similarity so largest similarities are first"""
        for k, v in self.sim_search_results.copy().items():
            self.sim_search_results[k] = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))

    def score_results(self):
        self.sim_search_results = self._convert_distance_to_similarity(nested_dict=self.sim_search_results)
        self._update_search_result_scores_for_missing_docs()
        self._sort_results_by_similarity_score()
        return self.sim_search_results

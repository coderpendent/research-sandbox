import random
import sys
from time import time
from collections import deque
from simhash import Simhash, SimhashIndex


def to_mb(x):
    return x / 1e6


"""
First a logic check: the deque and the index should always be in sync.
"""
nbr_docs = int(10)
max_cache = 5
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
index = SimhashIndex([], f=64, k=2)
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if (doc_id, simhash) not in tracker:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            print(f"removing {tracker[0][0]} from index to make room for {doc_id}")
            index.delete(*tracker[0])
        # add the new document
        print(f"appending {doc_id} to tracker")
        tracker.append((doc_id, simhash))
        print(f"appending {doc_id} to index")
        index.add(doc_id, simhash)
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")


"""
Now check that the max cache size is working.  Tracker should max out at 50 items.
"""
nbr_docs = 100
max_cache = 50
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
index = SimhashIndex([], f=64, k=2)
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if (doc_id, simhash) not in tracker:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            index.delete(*tracker[0])
        # add the new document
        tracker.append((doc_id, simhash))
        index.add(doc_id, simhash)
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")


"""
Now track the size of indexing many documents, and the time required.

Deque tracker was performing poorly so added a dict for lookups, while deque can track order.
"""
nbr_docs = 1_000
max_cache = 500_000
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
tracker_dict = dict()
index = SimhashIndex([], f=64, k=2)
start = time()
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if doc_id not in tracker_dict:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            index.delete(*tracker[0])
            del tracker_dict[tracker[0][0]]
        # add the new document
        tracker.append((doc_id, simhash))
        tracker_dict[doc_id] = simhash
        index.add(doc_id, simhash)
total_time = time() - start
print(f"\nTime to index {min(nbr_docs, max_cache)} documents: {total_time} seconds.")
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")
print(f"Dict of length {len(tracker_dict)} and size in Mb: {to_mb(sys.getsizeof(tracker_dict))}")

nbr_docs = 10_000
max_cache = 500_000
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
tracker_dict = dict()
index = SimhashIndex([], f=64, k=2)
start = time()
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if doc_id not in tracker_dict:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            index.delete(*tracker[0])
            del tracker_dict[tracker[0][0]]
        # add the new document
        tracker.append((doc_id, simhash))
        tracker_dict[doc_id] = simhash
        index.add(doc_id, simhash)
total_time = time() - start
print(f"\nTime to index {min(nbr_docs, max_cache)} documents: {total_time} seconds.")
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")
print(f"Dict of length {len(tracker_dict)} and size in Mb: {to_mb(sys.getsizeof(tracker_dict))}")

nbr_docs = 100_000
max_cache = 500_000
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
tracker_dict = dict()
index = SimhashIndex([], f=64, k=2)
start = time()
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if doc_id not in tracker_dict:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            index.delete(*tracker[0])
            del tracker_dict[tracker[0][0]]
        # add the new document
        tracker.append((doc_id, simhash))
        tracker_dict[doc_id] = simhash
        index.add(doc_id, simhash)
total_time = time() - start
print(f"\nTime to index {min(nbr_docs, max_cache)} documents: {total_time} seconds.")
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")
print(f"Dict of length {len(tracker_dict)} and size in Mb: {to_mb(sys.getsizeof(tracker_dict))}")

nbr_docs = 1_000_000
max_cache = 50_000_000
docs = [f"I {random.randint(1,nbr_docs)} to the {random.randint(1,nbr_docs)} store."] * nbr_docs
tracker = deque([], max_cache)
tracker_dict = dict()
index = SimhashIndex([], f=64, k=2)
start = time()
for doc_id, doc in enumerate(docs):
    # doc_id = 1  # uncomment to test what happens when an existing doc_id shows up
    simhash = Simhash(doc, f=64)
    # only proceed if the new document is not in the index
    if doc_id not in tracker_dict:
        # if new document would exceed the capacity of the index, remove the oldest
        if len(tracker) >= max_cache:
            index.delete(*tracker[0])
            del tracker_dict[tracker[0][0]]
        # add the new document
        tracker.append((doc_id, simhash))
        tracker_dict[doc_id] = simhash
        index.add(doc_id, simhash)
total_time = time() - start
print(f"\nTime to index {min(nbr_docs, max_cache)} documents: {total_time} seconds.")
print(f"Simhash index of {index.bucket_size()} buckets and size in Mb: {to_mb(sys.getsizeof(index))}")
print(f"Deque of length {len(tracker)} and size in Mb: {to_mb(sys.getsizeof(tracker))}")
print(f"Dict of length {len(tracker_dict)} and size in Mb: {to_mb(sys.getsizeof(tracker_dict))}")


"""
So the size of the Simhash Index depends on how much similarity there is (how many buckets the index needs).
These docs are nearly identical, so the index always has 3 buckets.  Need to test more.

A dictionary performed much faster than a deque, due to the need to check if an item exists in the tracker.  
Dict lookups are faster.  The drawback is higher memory.  Dict requires 5 Mb of memory, while deque requires < 1 Mb 
for 100k documents.

I could use a generator or parallelization to speed up the loop, but the API is designed to index & query 1 document 
at a time.  So the onus for efficiency is on the user.
"""
1.  Why are vector databases preferred over traditional relational databases for large-scale information retrieval in RAG systems, especially for semantic search?
2.  Explain the main scalability issue with k-nearest neighbor (KNN) search and how Approximate Nearest Neighbor (ANN) algorithms address this issue.
3.  Describe the core concept behind a hierarchical proximity graph in HNSW and how it contributes to faster search times.
4.  What is the primary purpose of "chunking" in a RAG system, and name two reasons why it's beneficial.
5.  How do overlapping chunks mitigate some of the problems associated with fixed-size chunking strategies?
6.  Briefly explain the mechanism of semantic chunking and its main advantage over fixed-size or recursive character splitting.
7.  What is "query rewriting" in the context of RAG systems, and how does an LLM typically facilitate this?
8.  Describe the main difference between a bi-encoder and a cross-encoder architecture for semantic search in terms of how they process prompts and documents.
9.  What is the primary trade-off when using ColBERT compared to a standard bi-encoder, in terms of system resources?
10. Explain the role of "re-ranking" in a RAG pipeline, specifically mentioning why more computationally expensive models (like cross-encoders) can be used for this step.
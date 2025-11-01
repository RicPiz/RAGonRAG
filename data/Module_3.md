# Optimizing Information Retrieval in RAG Systems
## Introduction
This briefing document provides a comprehensive overview of advanced techniques and considerations for optimizing information retrieval within Retrieval-Augmented Generation (RAG) systems, moving from theoretical foundations to production-grade implementations. It synthesizes key concepts from vector databases, approximate nearest neighbors algorithms, document chunking, query parsing, and re-ranking, all crucial for building efficient and effective RAG pipelines.

## Vector Databases and Scalable Retrieval
### The Need for Vector Databases
Traditional relational databases struggle with the scale and computational demands of semantic search, particularly when dealing with millions or billions of documents and complex vector operations.

> **Quote:** "You could use a traditional relational database to implement most of the retrieval techniques you just saw. But once you need to search millions or even billions of documents, some of those operations, in particular, the vector operations underlying semantic search will flow down significantly. At this point, you'll probably wanna switch over to a vector database."

Vector databases are specialized databases designed to store and search through vast quantities of high-dimensional vector data, making them almost synonymous with RAG systems. They are optimized for tasks such as building proximity graphs and computing vector distances, essential for vector-based applications.

### Approximate Nearest Neighbors (ANN) Algorithms
The simplest form of vector retrieval, k-nearest neighbor (k-NN) search, calculates the distance between a query vector and every document vector, then sorts documents by distance.

**Problem:** K-NN scales poorly, with computational requirements growing linearly with the number of documents. A billion documents would mean a billion vector distance calculations, making it impractically slow for large-scale systems.

Approximate Nearest Neighbors (ANN) algorithms provide a more efficient solution by using clever data structures to significantly speed up searches.

**Trade-off:** ANN algorithms make a "small sacrifice in the quality of results," meaning they are not guaranteed to find the absolute closest documents but will find ones that are "very close by."

**Pre-computation:** The computational intensity of building a proximity graph (the underlying data structure for many ANN algorithms) can be managed by pre-computing it before any prompts are received.

#### Navigable Small World (NSW)
- **Proximity Graph:** Before searching, NSW creates a proximity graph where each document is a node and edges connect a document to its closest neighbors.
- **Search Process:**
  - A query vector is created from the prompt.
  - The algorithm starts at a randomly chosen "candidate vector" (node).
  - It traverses the graph by repeatedly moving to the neighbor of the current candidate that is closest to the query vector.
  - The process stops when no neighbor is closer than the current candidate, and that candidate is returned.
- **Limitation:** It doesn't guarantee the absolute best match but finds very close vectors much faster than k-NN.

#### Hierarchical Navigable Small World (HNSW)
HNSW improves upon NSW by introducing a hierarchical proximity graph with multiple layers.

- **Layered Structure:**
  - **Layer 1:** Contains all vectors (e.g., 1,000 documents).
  - **Layer 2:** Contains a random subset of vectors (e.g., 100 documents) with its own proximity graph.
  - **Layer 3:** Contains an even smaller random subset (e.g., 10 documents) with its own proximity graph.
- **Search Process:** Searches begin at the top layer (sparsest graph) to quickly make "big jumps" towards the approximate neighborhood of the query vector. The search then progressively drops down to lower layers, refining the candidate vector, until the lowest layer (Layer 1) where the final best candidate is found.
- **Efficiency:** HNSW runtime is approximately logarithmic, significantly faster than KNN's linear runtime, enabling vector search at the scale of billions of vectors with low latency.

**Key Features of ANN:**
- Significantly faster than k-nearest neighbors at scale.
- Finds close documents, though not always the absolute best matches.
- Relies on a pre-computed proximity graph.

### Common Vector Database Operations (e.g., Weaviate)
Vector databases like Weaviate offer functionalities for setting up, loading, and searching data.

- **Setup:** Creating a database instance and defining collections to hold data, specifying data types (e.g., text) and the embedding model/vectorizer for semantic vectors.
- **Data Ingestion:** Adding data to collections, often in batches, with automatic error tracking. During this process, sparse vectors for keyword search and dense embedding vectors for semantic search are created, and an ANN index (like HNSW) is built.
- **Search Types:**
  - **Vector Search:** Querying with text and receiving results based on vector distance, along with metadata requests (e.g., distance between query and object vectors).
  - **Keyword Search (BM25):** Vector databases automatically create inverted indexes, allowing for efficient keyword searches (e.g., BM25).
  - **Hybrid Search:** Combines keyword and vector search, performed in parallel. An alpha parameter (e.g., 0.25) weighs the scores from both, balancing "semantic similarity of vector search and the strict matching similarity of keyword search." This is commonly used in production.
- **Filtering:** Applying filters based on metadata properties to narrow down results.

## Optimizing System Performance with Advanced Techniques
### Document Chunking
Chunking is the practice of breaking longer text documents into smaller, manageable text chunks. It's crucial for:

- **Embedding Model Limits:** Many embedding models have limits on the amount of text they can process.
- **Search Relevancy:** Smaller chunks provide a "sharper representation of any specific topic," improving search relevance. Vectorizing an entire book, for example, would result in a single vector that "kind of averages across all of them."
- **LLM Context Window:** Ensures only the most relevant text is sent to the LLM, preventing the context window from being filled by entire documents.

#### Chunk Size Considerations
- **Too Big:** Chunks at a chapter level might still be too large to capture nuanced meaning and will fill the LLM's context window.
- **Too Small:** Chunks at the word or sentence level lose surrounding context, diminishing search relevance.
- **Balance:** The goal is to find a balance between capturing too much or too little context.

#### Chunking Strategies
- **Fixed-Size Chunking (with Overlap):**
  - **Mechanism:** Chunks are of a predetermined size (e.g., 250 characters).
  - **Overlap:** To address arbitrary splits, chunks often overlap (e.g., 25 characters for a 250-character chunk). This "minimizes instances where words are cut off from their context" and increases the odds of relevant context appearing in multiple chunks.
  - **Starting Point:** "Fixed size chunks of about 500 characters with an overlap of 50 to 100 characters" is a good general starting point.
- **Recursive Character Text Splitting:**
  - **Mechanism:** Splits documents based on specific characters (e.g., newline character between paragraphs).
  - **Benefit:** Accounts for document structure, increasing the chance that "related concepts are kept together within a single chunk."
  - **Flexibility:** Different document types (HTML, code, plain text) can be split using different characters or tags.
- **Semantic Chunking:**
  - **Mechanism:** Groups sentences with similar meaning into chunks. It vectorizes the current chunk and the following sentence, adding the sentence if their vector distance is below a threshold. A new chunk starts when dissimilarity crosses the threshold.
  - **Benefit:** Creates variably sized chunks that follow the "train of thought of the author," placing splits at appropriate conceptual locations.
  - **Cost:** "Computationally expensive because you're repeatedly calculating vectors for every sentence in your knowledge base."
- **LLM-Based Chunking:**
  - **Mechanism:** An LLM is given the document and instructions (e.g., "separate chunks based on meaning, keeping similar concepts together... when new topics are discussed"). The LLM then generates the chunked output.
  - **Benefit:** "Very high-performing chunking strategy" due to the LLM's understanding of content.
  - **Cost:** Inherently a "black box approach" but becoming more economically viable as LLM costs decrease.
- **Context-Aware Chunking:**
  - **Mechanism:** An LLM is used to add summary text or context to each chunk, explaining its role in the broader document.
  - **Benefit:** The added text improves search relevancy (when the chunk is vectorized) and helps the LLM understand the chunk's overall meaning when retrieved.
  - **Cost:** "Computationally expensive pre-processing since an LLM needs to go through your entire knowledge base, one document, and chunk at a time to add context." No impact on search speed.
  - **Recommendation:** Often a good "first improvement to explore beyond fixed-width techniques" as it can be applied on top of any strategy.

### Query Parsing
Cleaning and transforming user-submitted prompts before submission to the retriever. This addresses the conversational nature of user input, which is often not optimized for direct database search.

**Benefit:** Optimizes the prompt for retrieval by clarifying intent, adding relevant terminology, and removing noise.

#### LLM-Based Query Rewriting (Most Widely Used)
- **Mechanism:** An LLM is prompted to rewrite the user's query.
- **Instructions:** The prompt to the LLM can include instructions like "Clarify ambiguous phrases, use medical terminology where applicable, add synonyms that increase the odds of finding matching document, remove unnecessary or distracting information."
- **Example:** A patient's prompt "I was out walking my dog... Three days later, my shoulder is still numb and my fingers are all pins and needles. What's going on?" could be rewritten to "Experienced a sudden forceful pull on the shoulder resulting in persistent shoulder numbness and finger numbness for three days. What are the potential causes or diagnoses such as neuropathy or nerve impingement?"
- **Benefit:** Substantial improvements in retrieval, easily justifying the additional LLM call cost.

#### Named Entity Recognition (NER)
- **Mechanism:** Identifies categories of information (people, places, dates, etc.) within the query.
- **Benefit:** This information can inform vector search or metadata filtering, significantly improving retrieval quality.
- **Example:** A model like Gliner can label entities within a query, even if it adds "a little bit of added latency."

#### Hypothetical Document Embeddings (HIDE)
- **Mechanism:** An LLM generates a "hypothetical document" that would be the ideal result for the search query. This hypothetical document is then embedded, and its vector is used for the actual search.
- **Benefit:** Helps the retriever understand not just the prompt's intent but also "what a high quality result would look like." It transforms the "apples to oranges" matching of prompts to documents into a "more similar text" comparison.
- **Cost:** Adds latency and computational resources for the LLM call.

### Advanced Retrieval Architectures: Cross-encoders and ColBERT
#### Bi-encoders (Vanilla Architecture)
- **Mechanism:** Each document and the prompt are embedded separately into a single semantic vector. An ANN algorithm then finds similar vectors.
- **Benefit:** All documents can be embedded ahead of time, "significantly speeding up search" since only the prompt needs to be embedded at query time.
- **Default:** "The default architecture for semantic search" due to reasonable quality, great speed, and minimal vector storage.

#### Cross-encoders
- **Mechanism:** Concatenates the document and the prompt, then passes the combined text into a specialized embedding model (essentially a reranker). This model directly outputs a relevancy score (0-1).
- **Benefit:** "Almost always provide better search results" than bi-encoders because they "understand deep contextual interactions between the prompt and document text." They offer the "gold standard" for search quality.
- **Problem:** "Scale terribly." For millions or billions of documents, billions of document-prompt pairs would need to be processed for each query. No pre-processing is possible since the prompt is needed.
- **Usage:** Too inefficient for default search but "a great tool for improving the results of other search techniques" (e.g., re-ranking).

#### ColBERT (Contextualized Late Interaction over BERT)
- **Mechanism:** Aims to combine the speed of bi-encoders with the quality of cross-encoders.
- **Document Embedding:** Each token in a document is embedded into its own semantic vector (not just the whole document).
- **Prompt Embedding:** Each token in the prompt is also embedded into its own semantic vector.
- **Scoring:** For each prompt token, the algorithm finds its most similar token in the document. These maximum similarity scores (max sim score) are summed to get an overall relevancy score for the document.
- **Benefit:** Provides "both the scalability of a bi-encoder and much of the rich interaction between prompts and documents found in a cross-encoder." Faster than cross-encoders, allowing for real-time use.
- **Drawback:** "The number of vectors you need to store increases proportionally to the tokens in both prompts and documents." A 2000-token document requires storing 2000 vectors, compared to a single vector in a bi-encoder.
- **Use Cases:** Increasingly supported by vector databases for projects requiring high precision and deep contextual understanding, such as "legal or medical fields," where the trade-off of increased storage for quality is acceptable.

### Re-ranking
A post-retrieval process that improves search quality by re-scoring and re-ranking an initial set of documents returned by the vector database.

- **Process:**
  - The vector database initially retrieves a larger set of documents (e.g., 20-100) using a faster search technique (e.g., hybrid search).
  - A re-ranker (typically a cross-encoder or LLM-based model) then re-scores these handful of documents based on a deeper understanding of their relevance to the prompt.
  - The documents are then re-ranked, and a smaller subset (e.g., 5-10) of the most relevant documents is returned.
- **Benefit:** Enables the use of high-performing but computationally expensive models (like cross-encoders) because they only operate on a limited number of documents, making the trade-off between quality and time feasible. It ensures "the absolute most relevant documents are returned."
- **Cost:** Adds "a little bit of latency to your overall system," but this trade-off is "almost always worth it."
- **LLM-based Re-ranking:** Similar to cross-encoders, an LLM assesses the relevance of prompt-document pairs and provides a numerical score. While promising, it has similar efficiency limitations to cross-encoders, making it suitable only as a re-ranking technique.
- **Implementation:** Often "quite easy to implement" (sometimes a single line in a search query) and provides a "great relevance boost."
- **Recommendation:** "One of the first techniques you should explore adding to your RAG pipeline when trying to improve search relevance."

## Conclusion
Building robust RAG systems involves a strategic combination of specialized tools and advanced techniques. Vector databases, powered by ANN algorithms, provide the scalable foundation for storing and retrieving vector data. However, optimizing retrieval performance requires further refinement through intelligent document chunking, precise query parsing, and sophisticated re-ranking strategies. While each technique comes with its own trade-offs in terms of computational cost and complexity, understanding these options allows RAG system designers to make informed decisions to achieve optimal search relevance and efficiency for their specific applications.


## Glossary of Key Terms
- **Approximate Nearest Neighbors (ANN) Algorithms:** A family of algorithms designed to find vectors that are very close to a query vector significantly faster than K-Nearest Neighbor (KNN) search, by sacrificing a small amount of accuracy.
- **Bi-encoder:** An architecture for semantic search where documents and prompts are embedded separately into single dense vectors. Document embeddings are pre-computed, enabling fast retrieval.
- **BM25:** A ranking function used by search engines to estimate the relevance of documents to a given search query, often used for keyword search.
- **Chunking:** The process of breaking longer text documents into smaller, more manageable pieces (chunks) before they are embedded and stored in a vector database.
- **ColBERT (Contextualized Late Interaction over BERT):** A semantic search architecture that generates a dense vector for each token in both documents and queries, allowing for deep, token-level interactions during scoring. Offers a balance between bi-encoder speed and cross-encoder quality.
- **Context-aware Chunking:** An advanced chunking technique where an LLM is used to add summary text or contextual information to each chunk, explaining its relevance within the broader document.
- **Cross-encoder:** An architecture for semantic search that concatenates the prompt and document text and processes them together to directly output a relevancy score. Provides high quality but is computationally expensive and slow for large datasets.
- **Dense Vectors:** Numerical representations (embeddings) of text that capture semantic meaning.
- **Embedding Model (Vectorizer):** A machine learning model that transforms text (or other data) into dense vectors.
- **Hierarchical Navigable Small World (HNSW):** A sophisticated ANN algorithm that builds a multi-layered proximity graph, enabling very fast, logarithmic-time searches by starting at higher, sparser layers and progressively refining the search in lower, denser layers.
- **Hybrid Search:** A search technique that combines both keyword search (using sparse vectors and inverted indexes) and semantic search (using dense vectors and ANN algorithms) to leverage the strengths of both.
- **Hypothetical Document Embeddings (HIDE):** A query parsing technique where an LLM generates a "hypothetical" ideal document that would answer the user's query. The vector of this hypothetical document is then used for the actual search.
- **K-Nearest Neighbor (KNN) Search:** The simplest form of vector retrieval, where the distance between a query vector and every document vector is calculated, and the k closest documents are returned. It scales poorly with large numbers of documents.
- **Knowledge Base:** The collection of documents or information that a RAG system retrieves from.
- **LLM-based Chunking:** An advanced chunking technique where a Large Language Model is instructed to generate chunks based on specific criteria (e.g., separating by meaning or topic).
- **Max Sim Score:** In ColBERT, the sum of the highest similarity scores between each prompt token and its most similar token in a document, used to determine document relevance.
- **Metadata:** Additional descriptive information associated with documents or chunks (e.g., author, date, source, location within a document).
- **Named Entity Recognition (NER):** A query parsing technique that identifies and categorizes specific entities (e.g., persons, locations, dates, organizations) within a user's query.
- **Navigable Small World (NSW):** An ANN algorithm that constructs a "proximity graph" where nodes represent documents and edges connect close neighbors, allowing for efficient traversal to find similar documents.
- **Overlapping Chunks:** A chunking strategy where consecutive chunks share a portion of their text, helping to maintain context across chunk boundaries.
- **Proximity Graph:** A data structure used by ANN algorithms (like NSW and HNSW) where documents are nodes and edges connect documents that are close to each other in vector space.
- **Query Parsing:** The process of analyzing and transforming a user's raw prompt into a more effective search query, often involving techniques like rewriting or entity recognition.
- **Query Rewriting:** A query parsing technique, typically using an LLM, to clarify, refine, or expand a user's prompt to optimize it for retrieval.
- **RAG (Retrieval-Augmented Generation) System:** An AI system that combines information retrieval with large language models. It retrieves relevant documents or chunks from a knowledge base and uses them to ground the LLM's generation, improving accuracy and relevance.
- **Re-ranking:** A post-retrieval process where an initial set of retrieved documents is re-scored and re-ordered using a more sophisticated (and often more computationally expensive) model to ensure the most relevant results are ultimately returned.
- **Recursive Character Text Splitting:** A chunking strategy that splits documents based on specific characters (e.g., newline, period, paragraph tags), attempting to preserve structural coherence.
- **Relational Database:** A traditional database that stores and organizes data in tables with predefined relationships. Not optimized for high-dimensional vector search.
- **Semantic Chunking:** An advanced chunking technique that groups sentences into chunks based on their semantic similarity, splitting when the meaning diverges significantly.
- **Semantic Search:** A search method that understands the meaning and context of a query, rather than just matching keywords, by comparing the dense vector embeddings of queries and documents.
- **Sparse Vectors:** Numerical representations used in keyword search, often reflecting word frequency or presence, that are mostly zeros.
- **Vector Database:** A specialized database optimized for storing, indexing, and searching high-dimensional vector data, specifically designed for tasks like semantic search and ANN algorithms.
- **Weaviate:** A popular open-source vector database used as an example in the course.
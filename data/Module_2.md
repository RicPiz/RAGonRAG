# RAG System Retriever Principles and Hybrid Search Strategies

## Overview

This document provides a detailed review of the core principles and techniques employed in a retriever component of a Retrieval Augmented Generation (RAG) system. A retriever's primary function is to efficiently locate relevant documents from a knowledge base to help a Large Language Model (LLM) formulate a response to a user's prompt. This process is complex due to the informal nature of user queries and the often unstructured format of knowledge base documents. Modern retrievers typically utilize a "hybrid search" approach, combining multiple techniques to achieve optimal performance.

## Key Themes and Concepts

### 1. The Retriever's Role and Challenges

The retriever faces a significant challenge: to *"find documents in the knowledge base that can help an LLM respond to a prompt."* Users interact with LLMs naturally, not with structured queries, while knowledge bases contain *"messily structured information"* such as emails, memos, or journal articles. The retriever must *"rapidly return the most relevant pieces, all in fractions of a second."*

### 2. Hybrid Search: The Core Strategy

Most modern retrievers employ a hybrid search strategy, which integrates three primary techniques:

-   Keyword Search
-   Semantic Search
-   Metadata Filtering

This combination leverages the strengths of each method to produce a comprehensive and accurate set of results.

### 3. Detailed Explanation of Search Techniques

#### 3.1 Metadata Filtering

-   **Functionality:** Metadata filtering uses *"rigid criteria stored in document metadata to narrow down search results."* This metadata can include information like *"a document's title, author, creation date, access privileges, and so forth."* It operates like filtering a spreadsheet or writing a SQL query, where only documents meeting all specified conditions are passed.
-   **Advantages:**
    -   **Simplicity:** *"conceptually simple, making it easy to understand how the system works and debug issues."*
    -   **Speed and Maturity:** *"fast, mature, and well-optimized approach."*
    -   **Strict Criteria:** *"the only approach that allows your system to decide whether documents are retrieved based on rigid criteria."* This is crucial for controlling access (e.g., paid subscribers vs. free articles) or regional relevance.
-   **Limitations:**
    -   It is *"not really a search technique"* on its own; it refines results from other methods.
    -   It is *"overly rigid, ignores a document's content, and lacks any way of ranking documents once they've passed the filter."*
    -   *"building a retriever that relied exclusively on metadata filtering would be essentially useless."* It *"needs to be paired with other search techniques."*

#### 3.2 Keyword Search

-   **Functionality:** Retrieves documents based on whether they *"share words in common with the prompt."* The core idea is that *"documents that contain a lot of words from the prompt are more likely to be relevant."*
-   **Bag of Words:** Both prompt and documents are treated as a *"bag of words,"* ignoring word order and focusing on word presence and frequency.
-   **Sparse Vectors/Inverted Index:** Word counts are stored in sparse vectors for each document, forming a *"term document matrix"* or *"inverted index,"* which facilitates quickly finding documents containing specific words.
-   **Scoring and Ranking (TF-IDF and BM25):**
    -   **TF-IDF (Term Frequency-Inverse Document Frequency):** A foundational method that scores documents by considering how often a keyword appears in a document (Term Frequency) and how rare that keyword is across the entire knowledge base (Inverse Document Frequency). Rare words contribute more to the score.
        -   **Formula:** The TF-IDF score is calculated as follows:
            $$
            \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
            $$
            Where:
            -   $ \text{TF}(t, d) $ = (Number of times term $t$ appears in document $d$) / (Total number of terms in document $d$)
            -   $ \text{IDF}(t, D) $ = $ \log$(Total number of documents $D$) / (1 + Number of documents with term $t$ in it)
        -   **Example:** Consider a document with 100 words where the word "AI" appears 5 times. The TF is $5/100 = 0.05$. If the word "AI" appears in 1,000 documents out of a total of 10,000,000, the IDF is $ \log(10,000,000 / 1000) = \log(10,000) = 4 $. The TF-IDF score for "AI" in this document is $0.05 \times 4 = 0.2$.

    -   **BM25 (Best Matching 25):** The *"algorithm used in most retrievers."* It improves on TF-IDF by:
        -   **Term Frequency Saturation:** Applying *"diminishing returns as they include more instances of a keyword,"* meaning 20 instances aren't necessarily twice as relevant as 10.
        -   **Document Length Normalization:** Penalizing longer documents less aggressively than TF-IDF, with *"diminishing additional penalties as documents grow in length."*
        -   **Tunable Hyperparameters:** Includes two parameters to control the degree of term frequency saturation and document length normalization, allowing for customization to specific datasets.
        -   **Formula:** The BM25 score for a document $D$ given a query $Q$ with terms $q_1, ..., q_n$ is:
            $$
            \text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
            $$
            Where:
            -   $ \text{IDF}(q_i) $ is the IDF of query term $q_i$.
            -   $ f(q_i, D) $ is the frequency of term $q_i$ in document $D$.
            -   $ |D| $ is the length of document $D$.
            -   $ \text{avgdl} $ is the average document length in the collection.
            -   $ k_1 $ and $ b $ are hyperparameters (typically $k_1 \in [1.2, 2.0]$ and $b = 0.75$).

-   **Advantages:**
    -   **Simplicity and Effectiveness:** A *"relatively straightforward approach that works well in practice."*
    -   **Exact Matching:** *"ensures that retrieved documents will contain the keywords from your user's prompt,"* which is vital for *"technical terminology or exact product names."*
    -   **Time-tested:** *"has powered retrieval in databases and search engines for decades."*
-   **Limitations:**
    -   It *"ultimately depends on the query containing keywords that exactly match the words in the document."*
    -   *"If a user sends a prompt that has a similar meaning to a document but just doesn't include the right words, keyword search won't be able to find that match."* (e.g., "happy" vs. "glad").

#### 3.3 Semantic Search

-   **Functionality:** Matches documents to prompts based on *"shared meaning"* by capturing nuances that keyword search misses (e.g., understanding synonyms, distinguishing homonyms).
-   **Embedding Models:** Documents and prompts are processed by an *"embedding model,"* which *"map words to a location in space"* represented by a vector.
-   **Vector Space:** Embeds *"semantically similar words to nearby locations in space,"* while dissimilar concepts are *"embedded farther apart."* These vectors typically have *"hundreds or even thousands of components."*
-   **Distance Measures:** Similarity between texts is quantified by measuring the distance between their vectors.
    -   **Cosine Similarity:** *"measures the similarity in the direction of two vectors,"* ranging from 1 (same direction) to -1 (opposite direction). Higher values indicate closer vectors/more similar concepts.
        -   **Formula:** For two vectors A and B:
            $$
            \text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
            $$
        -   **Example:** If vector A = [1, 2] and B = [3, 4], the cosine similarity is $ (1*3 + 2*4) / (\sqrt{1^2+2^2} * \sqrt{3^2+4^2}) = 11 / (\sqrt{5} * \sqrt{25}) = 11 / (2.236 * 5) \approx 0.9839 $. This high value indicates the vectors point in a very similar direction.

    -   **Dot Product:** Measures the *"length of the projection of one vector onto another."* Higher values for similar length and direction.
        -   **Formula:** For two vectors A and B:
            $$
            \text{Dot Product} = A \cdot B = \sum_{i=1}^{n} A_i B_i
            $$
        -   **Example:** If vector A = [1, 2] and B = [3, 4], the dot product is $ (1*3 + 2*4) = 11 $.

-   **Embedding Model Deep Dive (Training):**
    -   **Contrastive Training:** Embedding models are trained using *"positive and negative pairs"* of text. Positive pairs (e.g., "good morning" and "hello") should be embedded close, while negative pairs (e.g., "good morning" and "that's a noisy trombone") should be embedded far apart.
    -   **Iterative Adjustment:** The model starts with random vectors and iteratively updates its internal parameters to *"move positive pairs closer together and negative pairs further apart."*
    -   **Abstract Nature:** Semantic vectors are *"abstract and somewhat random"* initially, gaining meaning only through this training process. Different training runs with different initial random vectors will still form the same semantic clusters, but at different locations in vector space.
    -   **Model Specificity:** *"you only compare vectors generated by the same embedding model"* due to differences in training data, dimensions, and initialization.
-   **Advantages:**
    -   **Flexibility:** *"provides a flexibility that no other search technique can"* by matching based on meaning, not just exact words.
    -   **Nuance:** *"can capture nuances that keyword search misses."*
-   **Limitations:**
    -   **Computational Intensity:** It is *"slower and more computationally intensive than keyword search."*

## 4. Hybrid Search Pipeline

The hybrid search process orchestrates these techniques:

1.  **Prompt Reception:** The retriever receives the user's prompt.
2.  **Dual Search:** It simultaneously performs *"both a keyword search and a semantic search,"* generating two separate ranked lists of documents (e.g., 20-50 documents each).
3.  **Metadata Filtering:** *"Both of these lists are filtered using a metadata filter"* to remove irrelevant documents based on strict criteria (e.g., user department, access rights).
4.  **Rank Combination (Reciprocal Rank Fusion - RRF):** The two filtered lists are combined into a single, unified ranking using algorithms like Reciprocal Rank Fusion (RRF).
    -   **RRF Mechanism:** Documents are rewarded *"for being ranked highly on either list."* Points are awarded based on the reciprocal of their rank (e.g., 1st place = 1 point, 2nd place = 0.5 points).
        -   **Formula:** The RRF score for a document $d$ is calculated as:
            $$
            \text{RRF Score}(d) = \sum_{i \in \text{rank lists}} \frac{1}{k + \text{rank}_i(d)}
            $$
            Where:
            -   $ \text{rank}_i(d) $ is the rank of document $d$ in list $i$.
            -   $ k $ is a hyperparameter (default is often 60) that mitigates the impact of high ranks.
    -   **Hyperparameter K:** Controls the *"impact of the highest ranked documents."* A low K heavily favors top-ranked documents, while a higher K (e.g., 50) balances the influence across ranks.
    -   **Hyperparameter Beta:** Allows weighting the importance of semantic versus keyword search rankings (e.g., *"A 70-30 split, 70% semantic, 30% keyword search, is typically a good starting point"*).
5.  **Return Top K:** The retriever returns the *"top-ranked documents"* from this final hybrid list to be used by the LLM.

This hybrid approach allows a retriever to *"play to each approach's strengths and tune the system's performance to the data in your knowledge base or the needs of your overall project."*

## 5. Evaluating Retrieval Quality

Evaluating a retriever focuses on *"search quality,"* i.e., *"is it finding relevant documents?"* This requires:

-   The prompt itself.
-   The ranked list of documents returned by the retriever.
-   A *"ground truth list of all the relevant documents"* in the knowledge base.

Key metrics include:

-   **Precision:** The *"number of relevant retrieved documents by the total number of documents retrieved."* It *"penalizes a retriever for returning irrelevant documents and can be thought of as capturing how trustworthy the results are."*
    -   **Formula:**
        $$
        \text{Precision} = \frac{\text{Number of relevant documents retrieved}}{\text{Total number of documents retrieved}}
        $$
    -   **Example:** If a retriever returns 10 documents and 7 of them are relevant, the precision is $7/10 = 0.7$.
-   **Recall:** The *"number of relevant documents retrieved by the total number of relevant documents in the knowledge base."* It *"penalizes a retriever for leaving out any relevant documents and measures how comprehensive the retriever is."*
    -   **Formula:**
        $$
        \text{Recall} = \frac{\text{Number of relevant documents retrieved}}{\text{Total number of relevant documents}}
        $$
    -   **Example:** If there are 20 relevant documents in total and the retriever finds 7 of them, the recall is $7/20 = 0.35$.
-   **Trade-off:** Often, there's a trade-off where *"you're often trading off between the two."*
-   **Top K:** Metrics are often calculated *"in terms of the top K documents,"* meaning the highest-ranked K documents.
-   **Mean Average Precision (MAP) at K:** Evaluates the *"average precision for relevant documents in the first K documents retrieved."* It *"rewards ranking relevant documents highly,"* as irrelevant documents high in the ranking will decrease the overall average.
    -   **Formula:** MAP is the mean of the Average Precision (AP) scores over a set of queries.
        $$
        \text{AP@K} = \frac{1}{m} \sum_{k=1}^{K} P(k) \times \text{rel}(k)
        $$
        Where $m$ is the number of relevant documents, $P(k)$ is the precision at rank $k$, and $\text{rel}(k)$ is 1 if the document at rank $k$ is relevant, 0 otherwise.
    -   **Example:** If a search for a query returns 5 documents ranked `[R, NR, R, R, NR]` (R=Relevant, NR=Not Relevant), and there are 3 relevant documents in total.
        -   P@1 = 1/1 = 1
        -   P@2 = 1/2 = 0.5
        -   P@3 = 2/3 ≈ 0.67
        -   P@4 = 3/4 = 0.75
        -   P@5 = 3/5 = 0.6
        The AP@5 is $ (1*1 + 0*0.5 + 1*0.67 + 1*0.75 + 0*0.6) / 3 \approx (1 + 0.67 + 0.75) / 3 \approx 0.81 $. MAP would be the average of such AP scores over many queries.
-   **Mean Reciprocal Rank (MRR):** Measures the *"rank of the first relevant object in a returned list"* (e.g., if the first relevant document is at rank 2, reciprocal rank is 0.5). MRR *"reflects how soon on average you can find a relevant item in the retriever's ranking"* and emphasizes getting at least one relevant document high in the ranking.
    -   **Formula:**
        $$
        \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
        $$
        Where $|Q|$ is the number of queries and $\text{rank}_i$ is the rank of the first relevant document for the $i$-th query.
    -   **Example:** For three queries, if the first relevant document is found at ranks 2, 3, and 1 respectively, the MRR is $ (1/2 + 1/3 + 1/1) / 3 = (0.5 + 0.33 + 1) / 3 \approx 0.61 $.

### Metric Utility

-   **Recall at K:** Most foundational, capturing the *"most fundamental goal of the retriever, finding relevant documents."*
-   **Precision and MAP:** Assess the trustworthiness and effective ranking of relevant documents.
-   **MRR:** Specialized for evaluating performance at the very top of the ranking.
-   **Challenge:** All these metrics *"depend on having ground truth, relevant documents for a collection of sample prompts,"* which *"can be a time-consuming and manual process to compile."*

## Conclusion

The retriever is a critical component of RAG systems, tasked with the challenging job of intelligently sifting through vast, unstructured knowledge bases to find relevant information for LLMs. By combining the precision of keyword search (BM25), the nuanced understanding of semantic search (embedding models), and the strict control of metadata filtering, modern retrievers achieve a "hybrid search" capability. The performance of these systems can be rigorously evaluated using metrics like precision, recall, MAP, and MRR, allowing for continuous tuning and improvement to meet specific project needs.

## Glossary of Key Terms
- **Anchor Point**: In contrastive training, the primary piece of text for which similar (positive) and dissimilar (negative) examples are identified and used to adjust vector positions.
- **Bag of Words**: A text representation model where the order of words is ignored, and only the frequency of each word in a document matters. Used in keyword search.
- **Best Matching 25 (BM25)**: A ranking function used in keyword search that improves upon TF-IDF by incorporating term frequency saturation and diminishing penalties for document length, offering tunable hyperparameters.
- **Contrastive Training**: A method for training embedding models that uses pairs of similar (positive) and dissimilar (negative) texts to iteratively adjust vector positions, pulling positive pairs closer and pushing negative pairs farther apart.
- **Cosine Similarity**: A measure of similarity between two vectors that calculates the cosine of the angle between them. It ranges from -1 (opposite direction) to 1 (same direction), regardless of vector magnitude.
- **Document Length Normalization**: A technique in keyword search (especially BM25) that adjusts document scores to account for varying document lengths, preventing longer documents from being overly favored simply because they contain more words.
- **Dot Product**: A mathematical operation that measures the length of the projection of one vector onto another. It indicates similarity in both direction and magnitude; higher values suggest greater similarity.
- **Embedding Model**: A special mathematical model that transforms pieces of text (words, sentences, documents) into dense numerical vectors (embeddings), where semantically similar texts are mapped to nearby locations in a high-dimensional space.
- **Euclidean Distance**: The straight-line distance between two points (vectors) in a multi-dimensional space, calculated using a generalization of the Pythagorean theorem.
- **Ground Truth**: In retriever evaluation, the manually identified and verified list of all truly relevant documents in the knowledge base for a given prompt, used as the correct answers to assess performance.
- **Hybrid Search**: A retrieval strategy that combines multiple search techniques (typically keyword search, semantic search, and metadata filtering) to leverage their individual strengths and provide more comprehensive and accurate results.
- **Hyperparameter**: A parameter whose value is set before the training process begins, rather than being learned during training. Examples in retriever systems include the 'K' and 'Beta' in RRF, and parameters in BM25.
- **Information Retrieval**: The science of searching for information in a document collection or a database, a core component of RAG systems.
- **Inverted Index (Term Document Matrix)**: A data structure used in keyword search that maps words to the documents in which they appear, making it efficient to find all documents containing a specific word.
- **Inverse Document Frequency (IDF)**: A component of TF-IDF that measures how rare a word is across an entire knowledge base. Rare words have a higher IDF score, giving them more weight in relevance calculations.
- **Keyword Search**: A retrieval technique that finds documents based on the presence and frequency of exact words (keywords) shared with the user's prompt.
- **Knowledge Base**: The collection of documents or text files that the retriever searches to find relevant information for the LLM.
- **Large Language Model (LLM)**: An artificial intelligence model trained on vast amounts of text data, capable of understanding and generating human-like text.
- **Mean Average Precision (MAP@K)**: An evaluation metric that calculates the average precision for relevant documents within the top K retrieved documents, averaged across multiple queries. It rewards ranking relevant documents highly.
- **Mean Reciprocal Rank (MRR)**: An evaluation metric that measures the average reciprocal of the rank of the first relevant document found across multiple queries. It emphasizes the importance of finding at least one relevant item high in the ranking.
- **Metadata Filtering**: A retrieval technique that narrows down search results by strictly including or excluding documents based on predefined criteria stored in their associated metadata (e.g., author, date, access rights).
- **Negative Pair**: In contrastive training, two pieces of text that have dissimilar meanings and should be embedded far apart in vector space.
- **Precision**: An evaluation metric calculated as the number of relevant retrieved documents divided by the total number of documents retrieved. It indicates the trustworthiness of the results.
- **Positive Pair**: In contrastive training, two pieces of text that have similar meanings and should be embedded close together in vector space.
- **Prompt**: The user's input or query given to the RAG system, which the LLM will respond to using retrieved information.
- **RAG (Retrieval Augmented Generation) System**: An AI system that combines a retrieval component (the retriever) with a generation component (the LLM) to provide more accurate and contextually relevant responses.
- **Recall**: An evaluation metric calculated as the number of relevant retrieved documents divided by the total number of relevant documents existing in the knowledge base. It indicates the comprehensiveness of the retrieval.
- **Reciprocal Rank Fusion (RRF)**: A commonly used algorithm in hybrid search to combine multiple ranked lists of documents into a single, unified ranking, rewarding documents that appear highly in any of the individual lists.
- **Retriever**: The component of a RAG system responsible for searching a knowledge base and identifying documents relevant to a user's prompt.
- **Semantic Search**: A retrieval technique that finds documents based on their shared meaning with the user's prompt, even if exact keywords are not present, by using embedding models and vector similarity.
- **Sparse Vector**: A vector (like those representing word counts in keyword search) where most of its components are zero, indicating the absence of most words from a vocabulary in a given text.
- **Term Frequency (TF)**: The number of times a specific word appears in a document, a basic component of keyword search scoring.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: A statistical measure used in keyword search to reflect how important a word is to a document relative to a corpus. It combines term frequency and inverse document frequency.
- **Term Frequency Saturation**: A concept in BM25 where the relevance score for a document increases with the number of times a keyword appears, but at a diminishing rate, meaning many occurrences don't proportionally increase relevance indefinitely.
- **Vector Space**: A mathematical space where each point is represented by a vector. In semantic search, texts are mapped to points in this high-dimensional space, with proximity indicating semantic similarity.
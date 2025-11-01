# Retrieval Augmented Generation (RAG)

## Purpose

This document provides a detailed overview of Retrieval Augmented Generation (RAG), a critical technique for enhancing the performance and accuracy of Large Language Models (LLMs). It synthesizes key concepts, architectural components, applications, and advantages of RAG as presented in the provided sources.

## 1. Introduction to Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is the most widely used technique for improving the quality and accuracy of a large language model's response. It addresses a fundamental limitation of LLMs: their knowledge is confined to the data they were trained on (e.g., public internet data). RAG enables LLMs to access and utilize "additional data" or "propriety data," such as an organization's internal documents, to answer questions with facts they were not originally trained on.

"The core idea of RAG is pairing classical search systems with the reasoning abilities of large language models." This approach allows LLMs to "answer questions with facts that it was not already trained on." This is analogous to a human needing additional information for a complex question: "First, you collect any necessary information, and then you reason over that information to develop your response." In RAG, this collection of information is called retrieval, and the subsequent reasoning and response generation is called generation.

## 2. Why RAG is Essential

LLMs, while "remarkable tools" capable of various tasks like answering questions, summarizing text, and generating code, operate by predicting the next most probable word based on their training data. This mechanism, while powerful, leads to several limitations that RAG effectively mitigates:

*   **Limited and Outdated Knowledge:** LLMs only know information from their training data. They cannot answer questions about "very recent event or some specialized information it hasn't previously seen." Retraining LLMs is "costly and time-consuming," making them struggle to keep up with rapidly changing information.
*   **Hallucinations:** When faced with questions outside their training data, LLMs can "give responses that sound right, but aren't actually true." This is because "LLMs are designed to generate probable text, not truthful text." RAG directly addresses this by providing relevant context.
*   **Lack of Specificity/Personalization:** General LLMs lack access to specific, private, or personalized information (e.g., company policies, individual user data, project-specific code).
*   **Inability to Cite Sources:** Without external information, LLMs cannot reliably cite the sources for their answers.

RAG makes LLMs "more useful and more accurate by giving them access to information that they might not have had when being trained." Andrew Ng highlights its prevalence, stating, "I think RAG may be the most commonly built type of LLM-based application in the world today."

## 3. Core Architecture of a RAG System

A RAG system fundamentally involves three main components:

*   **Large Language Model (LLM):** The generative component that reasons over information and produces responses.
*   **Knowledge Base:** A collection of trusted, relevant, and potentially private information (e.g., documents, databases).
*   **Retriever:** The component responsible for searching the Knowledge Base and finding information relevant to a user's prompt.

The workflow is as follows:

1.  **User Prompt:** A user submits a question or request to the RAG system.
2.  **Retrieval:** The prompt is "first routes to the retriever." The retriever queries the knowledge base, which is "practically speaking is just a database of useful documents," to find "documents that it determines are most relevant to the prompt." The retriever ranks documents by relevance, aiming to return "the most relevant information from the knowledge base to share with the LLM."
3.  **Augmented Prompt Creation:** The system then creates an "augmented prompt," which incorporates "information from the relevant documents into the original prompt." An example provided is: "answer the following question, why are hotels in Vancouver so expensive this coming weekend? Here are five relevant articles that may help you respond, and then insert text from the articles."
4.  **Generation:** "The augmented prompt is sent to the LLM and it generates a response." The LLM uses both its pre-trained knowledge and the provided context to formulate an accurate and grounded answer.

This "side route through the retriever" ensures a "higher likelihood the response is accurate, up-to-date, and context aware."

## 4. Key Advantages of RAG

The addition of the retriever provides several significant advantages:

*   **Access to Novel and Private Information:** RAG makes "information available to the LLM that might otherwise not be," including "company policies, a piece of personal information, or this morning's headlines."
*   **Reduced Hallucinations:** By "adding relevant information directly in the prompt," RAG "grounds the language model's responses and makes them less likely to create generic or misleading text," thereby reducing "the likelihood of hallucinations or misleading responses."
*   **Up-to-Date Information:** RAG makes it "much easier to keep LLMs up-to-date with rapidly changing information." Instead of costly retraining, one can "simply update the information in the knowledge base."
*   **Source Citation:** RAG systems "can add citation information to the augmented prompt," enabling the LLM to include "that information in its ultimate response," which "doesn't just ground the response but enables human readers to dig deeper and validate the generated text."
*   **Optimized LLM Focus:** RAG allows the LLM to "focus on this text generation." The retriever handles "fact-finding or filtering steps," ensuring "each component is assigned to work on the area of its greatest strength."

## 5. Applications of RAG

RAG has broad applicability across various industries and use cases:

*   **Code Generation:** By using a codebase as a knowledge base, RAG helps LLMs "generate correct code for a specific project" by providing "classes, functions, and definitions in the project, as well as the overall coding style."
*   **Customizing Chatbots for Companies:** Companies can build customer service or internal chatbots that are "grounded" in their "own products, policies, and communication guidelines," minimizing "generic or misleading responses."
*   **Healthcare and Legal Domains:** RAG is crucial in fields where "precision is imperative and the amount of niche and potentially private information is vast." Knowledge bases can include "legal documents from a particular case" or "recently published medical journals."
*   **AI-Assisted Web Search:** Modern search engines provide AI summaries of search results, effectively acting as "a RAG system whose knowledge base is the entire internet."
*   **Personalized Assistants:** For applications like text messages, email clients, or word processors, RAG can leverage small-scale personal information (e.g., "your text messages, contact lists, emails, or a folder of documents") to "complete tasks in a way that is significantly more relevant to what you're doing."

## 6. Evolution and Future of RAG

RAG is a dynamic field, constantly evolving with advancements in LLM technology:

*   **Improved Grounding and Reduced Hallucinations:** Recent generations of models have shown significant improvements in "getting RAG systems to be more grounded in the documents or contexts it's given, so that over the last year or so, it feels like the hallucination rates of RAG systems have been steadily trending downwards."
*   **Handling Complex Questions:** Enhanced "reasoning models also let them tackle much more complex questions and can reason on top of the provided context."
*   **Larger Context Windows:** The increasing "input context window of LLMs" means that the best practices for "how do you cut documents into pieces to signal input context" are evolving, as "you don't need to squeeze so much information into a tiny little context window."
*   **Agentic Document Extraction:** Improvements in "agentic document extraction and related technologies" facilitate building RAG systems on "PDF files or slides or other types of documents," enabling them to "ingest and reason over and answer questions relating to broader sets of materials."
*   **Multi-Step Agentic Workflows:** RAG is increasingly integrated as "one component in a complex agentic workflow where maybe on, you know, step five or step seven of some internal enterprise workload, RAG gives the agent the information it needs."
*   **Agentic RAG:** This frontier development involves "building systems that use multiple large language models where each one handles a single part of a large workflow and has the agency to decide what data to retrieve." Instead of a human engineer manually determining retrieval rules, "you can give an AI agent tools to retrieve information and then let it decide, does it want to do a web search next? And if so, what keywords does it want to use for the web search? Or maybe query a specific specialized database." This self-correcting and flexible approach "gives them a way to deal with the messiness of the real world. If they mess up, they can route back and fix the approach that they're going with."

Even with rapid advancements in AI, "I don't think RAG's going anywhere. LLMs will continue to benefit from access to high-quality relevant data."

## 7. Understanding the Retriever and Information Retrieval

The retriever's function is to "provide useful information to the LLM that was potentially not available when the model was trained." It operates similarly to a librarian, understanding the meaning of a query to find relevant information within a vast collection.

Key aspects of the retriever:

*   **Knowledge Base and Indexing:** The retriever manages a "knowledge base of documents" and creates an "index of the documents... which keeps the documents organized and makes them easy to search."
*   **Query Processing and Similarity Scoring:** It "needs to process the prompt to understand its underlying meaning" and then uses that understanding to "search the index of documents." Documents are ranked by "relevance" through "numerical score[s] that quantifie[s] its relevance," often a "measure of the similarity between the text of the prompt and the text of the document."
*   **Relevance vs. Irrelevance:** A good retriever must not only return relevant documents but also "withhold irrelevant documents." Providing too much information can lead to "costly prompts or even entirely use up the LLM's context window." The challenge lies in determining the "exact right number" of documents to return.
*   **Optimization:** Optimizing retriever performance requires "monitor[ing] it over time and experiment[ing] with different settings."
*   **Vector Databases:** While not strictly necessary, "at scale, most retrievers will be built on top of a vector database, a specialized type of database that is optimized for rapidly finding the documents in your knowledge base that most closely match a prompt."

The field of "information retrieval was already mature when large language models first were developed, and ideas from this field underlie the way retrievers and RAG systems are designed."

This briefing document outlines the fundamental principles and practical implications of Retrieval Augmented Generation, emphasizing its role in making LLMs more accurate, relevant, and powerful across diverse applications.

### IV. Glossary of Key Terms
*   **Retrieval Augmented Generation (RAG):** A widely used technique for improving the quality and accuracy of a large language model's (LLM) response by providing it with access to additional, specific data it was not initially trained on.
*   **Large Language Model (LLM):** A powerful mathematical model of language, typically trained on massive datasets from the internet, capable of generating text, answering questions, summarizing, and more. LLMs predict the next most probable word (or token) in a sequence.
*   **Knowledge Base:** A collection of trusted, relevant, and potentially private or recent information (e.g., documents, databases, codebases) that a RAG system's retriever accesses.
*   **Retriever:** The component within a RAG system responsible for searching the knowledge base and finding the most relevant information or documents corresponding to a user's prompt.
*   **Augmented Prompt:** A prompt that has been modified to include not only the user's original question but also additional context and information retrieved from the knowledge base by the retriever.
*   **Hallucinations:** In the context of LLMs, these are instances where the model generates responses that sound plausible and grammatically correct but are factually incorrect or made-up, often due to a lack of relevant information in its training data.
*   **Generation (Phase):** The stage in a RAG system where the LLM processes the augmented prompt and produces a human-like text response.
*   **Retrieval (Phase):** The stage in a RAG system where the retriever actively searches and collects useful information from the knowledge base in response to a user's query.
*   **Context Window:** The maximum amount of text (measured in tokens) that an LLM can process or "see" at one time when generating a response.
*   **Tokens:** A more generic term than "words" for the pieces of text that LLMs process. Words, parts of words, and punctuation can all be tokens.
*   **Autoregressive:** Describes the behavior of LLMs where the choice of each new token generated is influenced by all the preceding tokens in the completion, making the text coherent.
*   **Parameters:** The numerical weights within an LLM's neural network that are adjusted during the training process, allowing the model to learn patterns in language.
*   **Grounded Responses:** LLM responses that are firmly based on factual information provided in the prompt, often through retrieval, which helps to prevent hallucinations and ensure accuracy.
*   **Vector Database:** A specialized type of database optimized for rapidly finding documents or data points that are "similar" to a given query, often used as the underlying technology for a retriever in production-scale RAG systems.
*   **Agentic RAG:** An advanced form of RAG where an AI agent (often an LLM itself) is given tools and the agency to dynamically decide what information to retrieve, how to retrieve it (e.g., web search, specialized database), and whether further retrieval steps are needed, enabling more flexible and powerful multi-step workflows.
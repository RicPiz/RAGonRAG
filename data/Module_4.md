# LLMs and RAG Systems: A Detailed Briefing

This briefing document provides a comprehensive overview of Large Language Models (LLMs) within the context of Retrieval Augmented Generation (RAG) systems, drawing insights from the provided source materials. It covers the foundational architecture of LLMs, strategies for controlling their behavior, factors in selecting an appropriate LLM, prompt engineering techniques, methods for handling hallucinations, evaluation strategies, the concept of agentic RAG, and a comparison with fine-tuning.

## 1. The Core Role of the LLM in RAG

While the Retriever component of a RAG system is crucial for finding and preparing information, **"the LLM is the real brains of the operation."** The Retriever provides the raw material, but it's the LLM that **"needs to actually use that information to generate a high-quality response."** This highlights the LLM's central role in understanding, synthesizing, and producing coherent and relevant output based on the retrieved context.

## 2. Understanding the Transformer Architecture

LLMs are built upon the Transformer architecture, originally proposed in the 2017 paper "Attention is All You Need."

### 2.1 Encoder-Decoder Structure

The original Transformer model had two main components:

*   **Encoder:** Processes original text to develop a deep contextual understanding.
*   **Decoder:** Uses this understanding to generate new text.

Most LLMs typically only include the second decoder component as their primary goal is text generation.

### 2.2 Prompt's Journey Through the Decoder

*   **Tokenization:** The prompt is split into individual "tokens."
*   **Initial Vector Representation:** Each token is assigned a static "first guess" dense vector representing its initial meaning.
*   **Positional Vectors:** Each token receives a positional vector indicating its location in the prompt.
*   **Attention Mechanism:** Tokens **"look at every other token in the prompt and can see both their meaning and their position."** They then decide **"which other tokens it should pay the most attention to,"** effectively determining which words have the biggest impact on their own meaning.
*   **Attention Heads:** Models utilize multiple "attention heads," each specializing in different types of relationships between words. These relationships are **"a complex and abstract set of relationships learned during the model's training."**
*   **Feedforward Phase:** After attention scores are assigned, the information enters the **"feedforward phase,"** the largest part of the LLM in terms of parameters. Here, **"updated vector embeddings for each token"** are assigned, representing a **"second guess of each token's true meaning, but now informed by the context of the other tokens in the text."**
*   **Iterative Refinement:** This process of attention and feedforward is typically repeated multiple times (e.g., 8 to 64 times), **"gradually refining its understanding at each stage."**
*   **Token Generation:** Based on the refined embeddings, the model calculates a probability distribution across its vocabulary for **"what tokens are likely to come next."** It then picks one token, weighted by its probability.
*   **Repetitive Generation:** To generate a full completion, **"the model does this process over and over again,"** incorporating previously generated tokens into the context for subsequent choices.

### 2.3 Implications for RAG Systems

Understanding the Transformer architecture highlights several key points for RAG:

*   **Why RAG Works:** LLMs **"are able to deeply understand the meaning and relevance of the information added to the prompt,"** thanks to the attention mechanism and feedforward layers.
*   **Inherent Randomness:** LLMs are **"still inherently random."** Even with relevant information, they **"may randomly choose not to generate text based on that information."** Controlling this randomness is crucial.
*   **Computational Cost:** Generating a single token is **"computationally expensive,"** and costs **"grow as the length of the prompt or completion does,"** impacting RAG system expenses.

## 3. LLM Sampling Strategies: Controlling Randomness

LLMs make **"a weighted random choice"** for every token added. Decoding and controlling this probability distribution is a big part of how you tune your LLM's behavior.

### 3.1 Greedy Decoding

*   **Description:** Always picks the token with the highest probability, eliminating randomness.
*   **Upside:** Makes the LLM deterministic, producing the same response for the same prompt. Useful for **"code completion or even as a temporary setting to debug your system."**
*   **Downside:** Can lead to **"predictable"** and **"generic or even stilted"** text, and the LLM can **"get stuck producing the same sequence of words over and over again."**

### 3.2 Temperature

*   **Description:** The **"most widely used parameter to control an LLM's randomness."** It acts like a dial, changing the shape of the probability distribution.
*   **Setting:**
    *   **Default (1):** Original distribution.
    *   **Lower (e.g., 0):** Leads to a **"more spiky distribution,"** with fewer tokens having a real chance (0 sets to greedy decoding).
    *   **Higher (e.g., 1.1-1.3):** **"Flattens out the probability distribution,"** giving less likely tokens a greater chance, leading to **"more variety and sometimes more interesting or even creative sounding text."**
*   **Too High:** Results in a **"very flat probability distribution,"** with all tokens having an equal chance, potentially generating **"nonsense tokens."**

### 3.3 Top K Sampling

*   **Description:** Limits the LLM to choosing from the **"top k most likely tokens,"** regardless of their individual probabilities.

### 3.4 Top P Sampling (Nucleus Sampling)

*   **Description:** Limits the LLM to choosing tokens **"with cumulative probability falling below some threshold."** It's more **"responsive or dynamic"** as it adjusts the pool size based on the distribution's shape (narrower pool for confident choices, larger for uncertain ones).

### 3.5 Other Techniques

*   **Repetition Penalty:** Decreases probabilities of words already appeared, making text **"more natural and varied."**
*   **Logit Biasing:** Permanently adjusts the probability of specific tokens up or down. Useful for:
    *   Suppressing undesirable words (e.g., profanity).
    *   Boosting desired categories for classification tasks.

### 3.6 General Advice

*   **"In general, I advise setting a temperature and top p that best fits your needs."**
*   **Lower temperature/top p:** For **"code or answering factual questions."**
*   **Higher temperature/top p:** For **"more creative domain."**
*   Consider repetition penalties and logit biases for specific issues. Experimentation is key.

## 4. Choosing Your LLM

Selecting the right LLM significantly impacts a RAG application's speed, quality, and budget.

### 4.1 Quantifiable Differences

*   **Model Size:** Measured in billions of parameters. Larger models (100-500B+) are generally more capable but **"always more expensive to run."**
*   **Cost:** Typically charged per million tokens (input/output). **"Newer, larger, and more capable models to cost more."**
*   **Context Window:** Maximum tokens an LLM can process (prompt + completion). Larger limits offer flexibility but still incur per-token cost.
*   **Speed (Time to First Token, Tokens per Second):** Critical for real-time interactions, potentially justifying lower performance in other areas.
*   **Training Cutoff Date (Knowledge Cutoff):** The last point in time represented in training data. A later cutoff is often preferred, **"especially in contexts where a model will need to respond to questions on recent events."**

### 4.2 Assessing Quality: LLM Benchmarks

Model quality, which includes reasoning ability and text readability, is harder to quantify. Benchmarks help compare models across various quality dimensions.

*   **Automated Benchmarks:** Score LLMs on tasks verifiable by code (e.g., multiple-choice tests, math/coding challenges).
    *   **Example:** MMLU (Massive Multitask Language Understanding) covers 57 subjects.
*   **Human Scoring:** Two anonymous LLMs respond to a prompt; human evaluators choose the preferred response.
    *   Results are ranked using an Elo algorithm.
    *   **Example:** LLM Arena. Captures **"nuanced quality factors that automated benchmarks can't easily measure."**
*   **LLM-as-a-Judge Benchmarks:** One LLM evaluates another LLM's responses against reference answers, providing a **"win rate."**
    *   **Upside:** Relatively cheap and flexible.
    *   **Downside:** Judge LLMs **"have a tendency to prefer answers from their own family of language models,"** requiring careful calibration.

### 4.3 Qualities of Good Benchmarks

*   **Relevant:** Aligns with the project's specific needs.
*   **Difficult:** Differentiates between high and low-performing models.
*   **Reproducible:** Scores are consistent across runs and verifiable.
*   **Align with Real-World Performance:** Benchmark scores should reflect practical utility.
*   **Data Contamination:** A potential issue where benchmark datasets might be included in LLM training data, leading to overperformance.

### 4.4 The Evolving Landscape

*   LLM performance on benchmarks rapidly improves, leading to saturation where most advanced models score near maximum. This necessitates the introduction of **"new and more challenging benchmarks."**
*   **"Models released today are usually significantly better than models from even a couple of years ago."** Expect to **"plan on eventually swapping in newly released models."**

## 5. Prompt Engineering: Building Your Augmented Prompt

**"To get the most out of your large language model, you'll need to write a high-quality prompt."** Prompt engineering encompasses techniques to achieve better results.

### 5.1 OpenAI's Messages Format

The most common format, structuring prompts as a JSON series of messages.

*   Each message has content (text) and a role (system, user, or assistant).
*   **System messages:** Influence overall behavior, high-level instructions.
*   **User messages:** User's current and previous prompts.
*   **Assistant messages:** LLM's previous responses.
*   In multi-turn conversations, the entire history is sent to the LLM with each new user prompt.

### 5.2 System Prompt

*   **Purpose:** Provides **"high-level instructions on how it should behave."**
*   **Content:** Can include tone, procedures, knowledge cutoff dates, current date, reasoning steps (e.g., "reason through answers step-by-step"), safety guidelines, and even a "personality."
*   **RAG-Specific Instructions:** Tell the LLM to **"use only the retrieved documents to answer prompts, or judge whether a document is relevant, or cite sources in its response."**
*   System prompts are typically added to every LLM call, making their refinement crucial for consistent quality.

### 5.3 Prompt Template for Augmented Prompts

A template defines the high-level structure and where content is injected. A typical augmented prompt includes:

*   High-level system prompt.
*   Previous messages (for multi-turn conversations).
*   Retrieved chunks (e.g., top 5 or 10) from the Retriever, with processing instructions.
*   Most recent user prompt.
*   Templates facilitate experimentation with different prompt structures.

## 6. Prompt Engineering: Advanced Techniques

Beyond basic templates, advanced techniques further enhance LLM performance.

### 6.1 In-Context Learning (Few-Shot/One-Shot Learning)

*   **Description:** Helps the LLM learn desired output by adding examples to the prompt.
*   **Application:** For a customer service chatbot, include examples of past requests and high-quality responses to guide the LLM's structure and tone.
*   **Implementation:**
    *   Hard-code examples for stable behaviors.
    *   Use RAG to retrieve relevant examples from a knowledge base for dynamic adaptation.

### 6.2 Reasoning-Oriented Strategies

Encourage LLMs to think step-by-step, improving accuracy and traceability.

*   **Think Aloud/Step-by-Step:** Tell the LLM to **"first think aloud or think step-by-step"** before providing a final answer, using a "scratchpad" (e.g., tokens between scratchpad tags).
*   **Chain of Thought Prompting:** Instructs the LLM to generate steps to answer a question, then follow them.
*   **Reasoning Models:** Many LLMs are now designed as **"reasoning models"** that inherently generate **"reasoning tokens"** (planning) before **"response tokens"** (final answer).
    *   **Pros:** Excel at complex tasks (coding, math, planning), more accurate.
    *   **Cons:** Typically **"slower and more expensive to run"** due to generating reasoning tokens.
*   **Note:** Many prompt engineering techniques (like "think step-by-step" or in-context learning) may not work as well with reasoning models, as they are already trained for this. Instead, focus on specific goals and desired output formats.

### 6.3 Context Window Management

Advanced techniques often increase prompt length, making context window management crucial.

*   **Single-Turn Conversations:** Validate if techniques add value; remove those that don't.
*   **Multi-Turn Conversations (Context Pruning):**
    *   Keep a fixed number of recent messages.
    *   Use a separate LLM to summarize older messages.
    *   Drop reasoning tokens from chat history if using a reasoning model, keeping only response tokens.
    *   Include only chunks retrieved for the most recent question in a RAG system.
*   **Longer Context Windows:** While helpful, long prompts remain **"slow and expensive to run."**

### 6.4 General Advice

*   **"A simple prompt template and a well-written system prompt might be all you need."** Add advanced techniques **"only after it's clear you need them."**
*   **"Prompting in general can be more of an art than a science. So whatever strategies you use, experiment with different prompts and find the ones that work best for your system."**

## 7. Handling Hallucinations

Hallucinations (generating plausible but inaccurate information) are a **"constant concern"** with LLMs, even in RAG systems.

### 7.1 Why LLMs Hallucinate

LLMs are **"designed to produce probable text sequences, with a bit of randomization thrown in for variety."** They **"aren't designed to differentiate between true and false, just probable and improbable."**

### 7.2 Problems with Hallucinations

*   Provide inaccurate information.
*   Sound plausible, making them difficult to detect.
*   Erode user trust over time.

### 7.3 Reducing Hallucinations in RAG

RAG is **"one of the best approaches currently available"** to minimize hallucinations by grounding responses in retrieved information. Additional steps are necessary:

*   **System Prompt Modification:** Instruct the LLM to **"only make factual claims based on retrieved information."**
*   **Source Citation:** Require the LLM to cite sources (e.g., at the end of each sentence/paragraph) to increase grounding and enable human verification.
    *   **Risk:** LLM might hallucinate citations.
*   **External Systems:** Tools like ContextCite can score how well a response is grounded, attributing sentences to source documents and labeling ungrounded statements as "no source."
*   **Benchmarks for Hallucination Detection:**
    *   **ALCE Benchmark:** Measures fluency, correctness, and citation quality by evaluating RAG systems against pre-assembled knowledge bases and questions.

### 7.4 General Advice

*   **"RAG is already taking the single most effective step to minimize hallucinations."**
*   Focus energy on **"ensuring the LLM grounds its answers and retrieved information by refining your system prompt."**
*   Test using **"hallucination-focused benchmarks"** for grounded, well-cited responses.

## 8. Evaluating Your LLM's Performance

Evaluating LLM performance is critical for making informed decisions about adjustments. It's important to focus on the LLM's specific role: **"to respond to the user prompt, incorporate the relevant information into its response, cite it appropriately, and resist getting distracted by any irrelevant information that was retrieved."**

### 8.1 LLM-as-a-Judge Metrics

Due to the subjective nature of LLM output quality, most metrics rely on other LLMs to assess responses.

*   **Ragas Library:** An open-source library providing RAG-specific metrics.
*   **Response Relevancy:** Measures whether a response is relevant to the user prompt (not necessarily factual accuracy). It works by using an LLM to generate sample prompts from the response, then comparing their semantic similarity to the original user prompt.
*   **Faithfulness:** Uses an LLM to identify factual claims in the response and then another LLM to determine how many are supported by retrieved information. The percentage of supported claims indicates faithfulness.
*   Other metrics assess sensitivity to irrelevant information and citation accuracy.
*   **Pattern:** **"Reliance on LLM calls at some point in the eval process and even possibly examples of ground truth correct answers."**

### 8.2 System-Wide Metrics

*   **User Feedback (A/B Testing):** Using user ratings (e.g., thumbs up/down) to A/B test changes to system prompts and observe impact on overall user satisfaction. This attributes performance changes to LLM settings.

### 8.3 General Advice

*   **"Plan on using either LLM as a judge-based evals or human feedback to assess LLM quality."**
*   A combination of techniques provides confidence in LLM performance.

## 9. Agentic RAG

**"Agentic workflows"** involve using **"several LLMs throughout your RAG system, each one responsible for a single step in the overall process."**

### 9.1 Key Changes in Agentic Systems

*   **Task Decomposition:** Tasks are treated as a series of steps and decisions, each completed by a different LLM call.
*   **Tool Access:** LLMs are given access to a wider array of tools (code interpreter, web browser, vector database).

### 9.2 Example Agentic Workflow

1.  **Router LLM:** A small, specialized LLM determines if the user's prompt requires vector database retrieval ("yes" or "no").
2.  **Conditional Path:** If "no," the prompt goes directly to a general LLM for response generation.
3.  **Retrieval and Evaluation:** If "yes," documents are retrieved. An Evaluator LLM determines if the retrieved documents are sufficient.
4.  **Iterative Retrieval:** If not sufficient, additional retrievals may be requested.
5.  **Response Generation:** Once sufficient, an augmented prompt is constructed and sent to a (potentially larger) LLM.
6.  **Citation LLM:** A final, specialized LLM adds citations to the response.

### 9.3 Key Points of Agentic Systems

*   **Flow Chart Design:** Design involves mapping out the workflow with each LLM completing a single task.
*   **Specialization:** Different LLMs can be used for different steps (e.g., lightweight models for routing/evaluation, larger models for generation, specialized models for citations).

### 9.4 Common Agentic Patterns

*   **Sequential Workflow:** Linear progression through a series of LLMs (e.g., query parser -> query rewriter -> citation generator).
*   **Conditional Workflow:** An LLM decides which of many paths a prompt should follow (e.g., router LLM for retrieval decision, or for choosing specialized LLMs).
*   **Iterative Workflow:** Routes the prompt back to an earlier point in the system, forming a loop (e.g., an evaluator LLM providing feedback on generated code until acceptable).
*   **Parallel Workflow:** An Orchestrator LLM breaks a prompt into multiple tasks, assigns them to separate LLMs, and a Synthesizer LLM recombines their work.

### 9.5 Building Agentic Systems

*   Simple systems can be self-implemented.
*   Complex systems benefit from **"a wide variety of tools, libraries, and platforms."**
*   **Mindset Shift:** LLMs become **"modular pieces that fit inside a larger workflow,"** enabling the use of smaller, specialized models.

## 10. RAG vs. Fine-Tuning

RAG and fine-tuning are distinct but complementary techniques for improving LLM performance.

### 10.1 Fine-Tuning

*   **Core Idea:** **"Retrain a language model with your own data to update its internal parameters."**
*   **Supervised Fine-Tuning (SFT):** Uses a labeled dataset (instructions/prompts and expected "ground truth" answers) to adjust model parameters.
*   **Instruction Fine-Tuning:** The dataset includes instructions (prompts/questions) and expected best answers.
*   **Domain Specialization:** Fine-tuning makes a model **"much more of an expert at answering that kind of question"** within a specific domain (e.g., healthcare, legal).
*   **Trade-offs:** Improves performance in the specialized domain but **"can actually decrease performance in other domains."**
*   **Small Models in Agentic Systems:** Highly effective for heavily fine-tuning small, lightweight models for single, discrete tasks (e.g., a router LLM).
*   **Limitation:** **"Fine-tuning is usually not a great way to teach an LLM new information."** Its impact is more on how the model responds (style, structure) than what information it knows.

### 10.2 RAG vs. Fine-Tuning: When to Use Which

*   **RAG Best for:** **"Knowledge injection."** If the LLM needs access to new information, RAG injects that information into the prompt.
*   **Fine-Tuning Best for:** **"Domain adaption."** If the LLM needs to specialize in a certain task or domain, fine-tuning is appropriate. This is particularly true for single, discrete tasks within an agentic system.

### 10.3 Combining RAG and Fine-Tuning

*   **"Fine-tuning and RAG can also be used together."**
*   **Example:** Fine-tune a model **"specifically to incorporate retrieved information into its final responses,"** essentially specializing its role within the RAG system.
*   **Conclusion:** **"The best choice might be both."** They are **"complementary tools,"** improving performance in different ways.

### 10.4 Practical Considerations

*   Fine-tuning is complex and often requires a dedicated course.
*   Pre-fine-tuned models are available in online repositories for specific tasks/domains.

This briefing provides a structured understanding of LLMs in RAG systems, from their underlying architecture to advanced optimization and evaluation techniques. The emphasis is on the LLM's central role, the iterative nature of its internal processing, the importance of controlling its inherent randomness, the practical considerations in choosing and prompting models, and the ongoing challenge of managing hallucinations. Finally, it introduces agentic workflows as a powerful design pattern and clarifies the complementary roles of RAG and fine-tuning.


## Glossary of Key Terms

**Attention Mechanism:** A core component of the transformer architecture that allows each token in a sequence to weigh the importance of all other tokens, developing a detailed understanding of their relationships and context.

**Agentic Workflow:** A system design where multiple Large Language Models (LLMs) collaborate, each specializing in a single step or decision within a complex task, often with access to external tools.

**ALCE Benchmark:** A benchmark designed to measure how well a system references and cites sources in its generated responses, evaluating metrics like fluency, correctness, and citation quality.

**Augmented Prompt:** A prompt constructed for a RAG system that includes not only the user's original query but also relevant information retrieved from a knowledge base, a system prompt, and potentially conversation history.

**Chain of Thought Prompting:** An advanced prompt engineering technique that instructs an LLM to tackle questions in a step-by-step manner, generating intermediate reasoning steps before providing a final answer.

**Context Pruning:** Techniques used to manage the context window in multi-turn conversations by reducing the length of older messages, for example, by keeping only recent messages or summarizing older ones.

**Context Window:** The maximum number of tokens an LLM can process in a single input, including both the prompt and the generated completion.

**ContextCite:** A system that scores how well an LLM's response is grounded in source materials by attributing each sentence to a specific retrieved document or labeling it as having "no source."

**Decoder:** A component of the transformer architecture (and most modern LLMs) responsible for generating new text based on an internal understanding of the input.

**Embeddings:** Dense vector representations of text (or tokens) that capture their semantic meaning, used by LLMs to process and understand language.

**Encoder:** A component of the original transformer architecture responsible for processing input text and developing a deep contextual understanding of its meaning.

**Faithfulness Metric:** An evaluation metric (e.g., from Ragas) that measures the percentage of factual claims made in an LLM's response that are actually supported by the information retrieved from the knowledge base.

**Feedforward Layer:** The largest part of an LLM, containing most parameters, which refines token embeddings based on their original meaning, position, and attention scores, gradually improving contextual understanding.

**Few-Shot Learning:** An in-context learning technique where multiple examples of desired input-output pairs are provided in the prompt to guide the LLM's generation.

**Fine-Tuning:** A technique that retrains an off-the-shelf LLM with a specific labeled dataset to update its internal parameters, specializing it for a particular domain or task.

**Greedy Decoding:** An LLM sampling strategy where the model consistently chooses the token with the highest probability at each step, making the output deterministic but potentially predictable.

**Hallucination:** When an LLM generates plausible-sounding but factually inaccurate or made-up information.

**In-Context Learning:** A prompt engineering technique where examples of desired outputs are added to the prompt to help the LLM learn the structure, tone, or type of response expected.

**Instruction Fine-Tuning:** A type of supervised fine-tuning where the dataset includes both input instructions (prompts/questions) and expected ground truth answers, used to adjust model parameters.

**LLM Arena:** A popular host for human-evaluated benchmarks, using an Elo algorithm to rank LLMs based on human preferences for responses to anonymous prompts.

**LLM-as-a-Judge:** A benchmark approach where one LLM is used to rate the responses of another LLM to test questions, often with access to reference answers, providing a "win rate" for comparison.

**Logit Biasing:** A sampling technique that permanently adjusts the probability of specific tokens being chosen by an LLM, increasing or decreasing their likelihood.

**MMLU (Massive Multitask Language Understanding):** An automated benchmark covering 57 subjects across various fields using multiple-choice tests to score LLM performance.

**One-Shot Learning:** A form of in-context learning where only one example of a desired input-output pair is provided in the prompt.

**Positional Vector:** A vector assigned to each token in a prompt that captures its location within the text, helping the LLM understand word order and relationships.

**Prompt Engineering:** An umbrella term for various techniques used to construct high-quality prompts to elicit better, more specific, or more controlled responses from an LLM.

**Prompt Template:** A pre-defined high-level structure for constructing prompts, indicating where different pieces of content (system prompt, retrieved context, user prompt) will be injected.

**Ragas Library:** An open-source library that provides RAG-specific evaluation metrics, often relying on LLMs to assess quality attributes like response relevancy and faithfulness.

**Reasoning Model:** An LLM designed to excel at complex reasoning tasks by first generating internal "reasoning tokens" (a scratchpad) before producing the final "response tokens."

**Repetition Penalty:** A sampling technique that decreases the probabilities of words or phrases that have already appeared in the LLM's completion, making the text sound more natural and varied.

**Response Relevancy Metric:** An evaluation metric (e.g., from Ragas) that measures whether an LLM's response is actually relevant to the user's original prompt, using semantic similarity between the original and generated prompts.

**Retrieval Augmented Generation (RAG):** A system that augments an LLM's prompt with relevant information retrieved from an external knowledge base to improve the quality, accuracy, and groundedness of its responses.

**Saturated Benchmark:** A benchmark where almost all advanced LLMs score near the maximum, indicating it no longer effectively differentiates between high and low-performing models, necessitating new, more challenging benchmarks.

**Self-Consistency Checking:** A method to detect hallucinations by having an LLM repeatedly generate completions for the same prompt and checking for factual inconsistencies across the outputs.

**Supervised Fine-Tuning (SFT):** The process of retraining an LLM using a labeled dataset, where the model's output is compared to correct answers, and its internal parameters are adjusted.

**System Prompt:** A message provided to the LLM at the beginning of a conversation or prompt to influence its overall behavior, tone, and the procedures it should follow.

**Temperature:** A parameter that controls the randomness or creativity of an LLM's token generation by adjusting the shape of the probability distribution for next token selection.

**Tokenization:** The process of splitting a text input (prompt) into smaller units called tokens, which are the fundamental units processed by an LLM.

**Tokens:** The individual units of text (words, subwords, characters) that an LLM processes.

**Top-K Sampling:** A sampling technique that limits the LLM's choice of the next token to only the k most probable tokens from the distribution.

**Top-P Sampling (Nucleus Sampling):** A sampling technique that limits the LLM's choice of the next token to the smallest set of most probable tokens whose cumulative probability exceeds a given threshold p.

**Transformer Architecture:** A neural network architecture introduced in 2017 ("Attention is All You Need") that revolutionized natural language processing, forming the basis of most modern LLMs due to its efficient use of attention mechanisms.
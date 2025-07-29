Of course, here is the full text in Markdown format.

-----

# University Regulations Q\&A Chatbot

This project is an intelligent chatbot that uses the RAG (Retrieval-Augmented Generation) technique to answer user questions about the university's educational regulations. Users can ask questions about academic rules, credits, probation, etc., and the bot provides accurate answers based on the regulations PDF file.

-----

## ✨ Features

  * **Accurate Responses:** Answers are generated directly based on the content of the regulations document.
  * **Natural Language Understanding:** Uses modern Large Language Models to effectively understand user questions.
  * **Web-based UI:** Features a simple and functional web-based user interface for easy interaction.
  * **Conversation History:** The bot can remember the context of the conversation within the same session.
  * **Source Citation:** Cites the source (e.g., page number) in its responses to increase credibility.

-----

## 🧠 AI Architecture

The project's intelligence is built on a two-phase **Retrieval-Augmented Generation (RAG)** pipeline.

### Phase 1: Indexing (Data Ingestion)

This is the preparatory phase, handled by the `ingest.py` script, which builds a searchable knowledge base from the source PDF.

1.  **Load & Chunk:** The regulations PDF is loaded page by page. The text of each page is then split into smaller, meaningful chunks based on the Persian keyword for "Article" (`ماده`).
2.  **Preserve Metadata:** As the document is chunked, crucial metadata—like the original page number—is preserved and attached to each chunk.
3.  **Embedding:** Each text chunk is converted into a numerical vector (an embedding) using the powerful **Cohere `embed-multilingual-v3.0`** model. This model is specifically chosen for its high performance on multilingual documents, including Persian.
4.  **Store:** These vectors are stored in a vector index on disk (in the `/storage` folder), creating a structured and efficiently searchable representation of the entire regulations document.

### Phase 2: Retrieval & Generation (Live Q\&A)

This phase occurs when a user asks a question in the web application.

1.  **Retrieval:** The user's question is first converted into a vector using the same Cohere model (this time in `search_query` mode for optimal relevance). The system then searches the vector index to find the text chunks with the most similar embeddings.
2.  **Augmentation:** The most relevant chunks, along with their source page numbers, are retrieved. This context is then inserted into a carefully engineered prompt that instructs the Large Language Model on its persona, rules, and task.
3.  **Generation:** The complete, augmented prompt is sent to the **Llama 3.1** model (via the Groq API). The model's job is not to answer from its general knowledge but to synthesize a precise and helpful answer based *only* on the factual context provided from the regulations document.

-----

## 🛠️ Tech Stack

  * **Programming Language:** Python
  * **Web Framework:** Flask
  * **AI Core:** LlamaIndex
  * **Generative LLM:** Llama 3.1 (via Groq API)
  * **Embedding Model:** Cohere `embed-multilingual-v3.0` (via Cohere API)

-----

## 🚀 Setup and Installation

Follow these steps to run the project locally.

### Prerequisites

  * Python 3.9+
  * Git

### 1\. Clone the Repository

First, clone the project from GitHub:

```bash
git clone https://github.com/Amir-Sa02/university-regulations-chatbot.git
cd university-regulations-chatbot
```

### 2\. Install Dependencies

Install all required libraries using the `requirements.txt` file. This file contains all the necessary Python packages for the project.

```bash
pip install -r requirements.txt
```

### 3\. Set Up Environment Variables

This project requires API keys to function. An example file named `.env.example` is included to show the required format.

```

Next, open the new `.env` file and add your actual API keys for the `GROQ_API_KEY` and `COHERE_API_KEY` variables.

### 4\. Add Data File

Place your educational regulations PDF file inside the `/data` folder.

### 5\. Running the Project

**Step 1: Ingest the Data**
First, you need to run the `ingest.py` script to read the PDF and build the vector database. (This is only required once initially or after the PDF file changes.)

```bash
python ingest.py
```

**Note:** Before running this command, make sure to delete the `storage` directory (if it exists) to build the new database.

**Step 2: Run the Web Application**
Now, run the web server:

```bash
python app.py
```

After it runs successfully, you can chat with the bot by navigating to `http://127.0.0.1:5000` in your web browser.
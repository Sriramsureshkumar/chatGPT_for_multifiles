
To respond to your inquiries, the application takes the following actions:


1. PDF Loading: The application reads various PDF files and extracts the text information from them.


2. Text Chunking: The retrieved text is broken into manageable-sized chunks for processing.


3. Language Model: To create vector representations (embeddings) of the text chunks, the application uses a language model.


4. Similarity Matching: After comparing your inquiry with the text chunks, the program shows you which ones are the most semantically related.


5. Response Generation: The language model receives the chosen chunks and produces a response based on the pertinent PDF material.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.


import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Function to chunk text into smaller parts
def chunk_text(text, max_length=1024):
    sentences = text.split('. ')
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    chunks.append(current_chunk.strip())
    return chunks

# Function to summarize long documents
def summarize_long_document(text, model_name='facebook/bart-large-cnn'):
    # Load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        # Tokenize the input text
        inputs = tokenizer(chunk, max_length=1024, return_tensors='pt', truncation=True)
        
        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return " ".join(summaries)

# Streamlit app layout
st.title("Intelligent Document Summarization Tool")

st.write("Enter the text you want to summarize in the box below:")

# Text area for input
input_text = st.text_area("Document Text", height=300)

# Summarize button
if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_long_document(input_text)
            st.subheader("Summary")
            st.write(summary)


import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- MODEL LOADING ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Loads the T5 model and tokenizer."""
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return tokenizer, model

tokenizer, model = load_model()

# --- SUMMARIZATION FUNCTION ---
def summarize_text(text):
    """Generates a summary for the given text using the T5 model."""
    # Prepend the "summarize: " prefix
    input_text = "summarize: " + text
    
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = model.generate(input_ids,
                                 max_length=150,
                                 min_length=40,
                                 length_penalty=2.0,
                                 num_beams=4,
                                 early_stopping=True)
    
    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- STREAMLIT UI ---
st.set_page_config(page_title="Text Summarizer", page_icon="✍️", layout="wide")

st.title("NLP Text Summarizer ✍️")
st.subheader("Summarize long text or articles into a few key sentences.")

# Text area for user input
st.markdown("### Enter the text you want to summarize:")
input_text_area = st.text_area("You can paste your text here:", height=250)

# Button to trigger summarization
if st.button("Generate Summary", type="primary"):
    if input_text_area:
        with st.spinner("Summarizing... This might take a moment. 🤔"):
            # Generate the summary
            summary_result = summarize_text(input_text_area)
            
            # Display the result
            st.markdown("### Here's your summary:")
            st.success(summary_result)
    else:
        st.warning("Please enter some text to summarize.")

st.markdown("---")
st.write("Built with Streamlit and Hugging Face Transformers.")

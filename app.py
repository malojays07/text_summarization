import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead

# Google Drive file ID
file_id = "https://drive.google.com/file/d/1WrJ2XKaWdaJ7qhThL0mWZ9sTGIkTcSHi/view?usp=sharing"  # Replace with the file ID of your Google Drive file

# Generate the direct download link
download_link = f"https://drive.google.com/uc?export=download&id={file_id}"

# Load the pre-trained model and tokenizer
model_name = file_id  # Replace with the name or path of your pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

# Function to generate the summary
def generate_summary(text):
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Streamlit app
def main():
    st.title("Text Summarization")

    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "docx"])

    # Generate summary button
    if st.button("Generate Summary"):
        if uploaded_file is not None:
            # Read uploaded file
            text = uploaded_file.read().decode("utf-8")
            # Generate the summary
            summary = generate_summary(text)
            st.success("Summary:")
            st.write(summary)
        else:
            st.warning("Please upload a document.")

# Run the app
if __name__ == "__main__":
    app()

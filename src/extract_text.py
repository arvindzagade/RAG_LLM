import fitz 
import os # PyMuPDF


def extract_text_from_pdfs(data_folder="data/"):
    texts = []
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(data_folder, file))
            text = "\n".join([page.get_text("text") for page in doc])
            texts.append({"filename": file, "text": text})
    return texts

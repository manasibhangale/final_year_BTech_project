import fitz  # PyMuPDF
import docx
import pytesseract
from bs4 import BeautifulSoup
import requests
from PIL import Image


def load_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def load_txt(file):
    return file.read().decode("utf-8")


def load_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)


def load_url(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text()


def load_document(file):
    if file.type == "application/pdf":
        return load_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return load_docx(file)
    elif file.type == "text/plain":
        return load_txt(file)
    elif "image" in file.type:
        return load_image(file)
    return ""
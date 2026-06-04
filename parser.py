from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar



for page_layout in extract_pages('your_document.pdf'):
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                print(text_line.get_text().strip())


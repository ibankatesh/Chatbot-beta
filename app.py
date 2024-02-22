from flask import Flask, render_template, request, send_file
import pandas as pd
import re
import numpy as np  
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from PyPDF2 import PdfReader 
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from concurrent.futures import ThreadPoolExecutor
import io

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Handle NaN values in 'Narration' column
def handle_nan(narration):
    if pd.isnull(narration):
        return ""
    return str(narration)

def extract_sender_receiver_name(narration):
    names = re.findall(r'\b(?:[A-Z]+\s)+[A-Z]+\b', handle_nan(narration))
    return names[0] if names else None 

def extract_payment_method(narration):
    narration = handle_nan(narration)
    if narration == "":
        return None
    methods = re.findall(r'(?:UPI|IMPS|NEFT|RTGS)\b', narration)
    return methods[0] if methods else None  

def extract_payment_platform(narration):
    narration = handle_nan(narration).lower()
    if 'phone pe' in narration or '@ybl' in narration or '@axl' in narration:
        return 'PhonePe'
    elif 'paytm' in narration:
        return 'Paytm'
    elif 'bharatpe' in narration:
        return 'Bharat Pay'
    elif 'cash wdl' in narration:
        return 'ATM'
    elif '@ok' in narration:
        return 'Google Pay'
    else:
        return None

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def nltk_named_entity_recognition(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    entities = ne_chunk(tagged)
    return entities

def process_pdf_file(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    entities = nltk_named_entity_recognition(text)
    return text, str(entities)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file, engine='openpyxl')
            elif file.filename.endswith('.pdf'):
                with ThreadPoolExecutor() as executor:
                    text, entities = executor.submit(process_pdf_file, file).result()
                df = pd.DataFrame({'Narration': [text], 'Named_Entities': [entities]})
            else:
                return 'Invalid file format'

            df['Name'] = df['Narration'].apply(extract_sender_receiver_name)
            df['Payment_Method'] = df['Narration'].apply(extract_payment_method)
            df['Payment_Platform'] = df['Narration'].apply(extract_payment_platform)

            return render_template('result.html', tables=[df.to_html(classes='data', header="true")], titles=df.columns.values, data=df.to_dict('records'))
        except Exception as e:
            return f'Error processing file: {str(e)}'

@app.route('/download', methods=['POST'])
def download_data():
    data = request.form['data']
    df = pd.DataFrame(eval(data))
    
    if request.form['format'] == 'csv':
        csv_data = df.to_csv(index=False)
        return send_file(io.BytesIO(csv_data.encode()), download_name="data.csv", as_attachment=True)
    elif request.form['format'] == 'pdf':
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        table_data = [[row['Name'], row['Payment_Method'], row['Payment_Platform']] for index, row in df.iterrows()]
        table = Table(table_data, colWidths=[200, 200, 200])
        style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)])
        table.setStyle(style)
        elements = [table]
        doc.build(elements)
        pdf_buffer.seek(0)
        return send_file(pdf_buffer, download_name="data.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

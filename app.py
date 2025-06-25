# import atau gunakan library flask untuk render template, request, dan jsonify 
from flask import Flask, render_template, request, jsonify
import re
import os
import json
import pickle
import pandas as pd

# proses modifikasi
# import fungsi dalam library nltk, yakni stopword dan lammatizer atau proses setemmingnya
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# proses modifikasi
# import fungsi-fungsi untuk query prepocessing dan class perhitungan bm25 pada file
from query_prepocessing import index_builder, convert_to_lemma
from bm25_calculation import BM25_OKAPI

# import library spacy yang digunakan untuk 
import spacy
# kemudian load model bahasa Inggris spaCy (bisa di-load sekali saja)
nlp = spacy.load('en_core_web_sm')

# proses modifikasi
# instance dari kelas flask untuk mmenjalankan framework flask
app = Flask(__name__)

# proses modifikasi
# Inisialisasikan setup stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# proses modifikasi
# load data indian_food
with open('indian_food.json', encoding='utf-8') as f:
    # kemudaian load data berupa json
    data = json.load(f)


# ubah load data pada indian food menjadi dataFrame
df = pd.DataFrame(data)
# proses modifikasi
# hitung kategori setiap resep masakan berdasarkan asal resep masakan 
total_category = df.groupby(['Cuisine'])['TranslatedRecipeName'].count().sort_values()

# proses modifikasi
# proses penggabungan data kolom pada satu value content
df['content'] = df['TranslatedRecipeName'] + " " + df['TranslatedIngredients'] + " " + df['TranslatedInstructions']

# proses mengcopy data untuk proses prepocessing
prepocessing_data = df.copy()
# mengubah data dari column content menjadi lowercase menggunkan fungsi .lower()
prepocessing_data['content'] = prepocessing_data['content'].apply(lambda x: x.lower())
# mengubah atau mengekstrak kalimat pada data agar bersih dari simbol dan emoticon
prepocessing_data['content'] = prepocessing_data['content'].apply(lambda x: ' '.join(re.findall(r'\w+', x)))

# proses membuat folder jika belum ada
if not os.path.exists('./saved_obj'):
    os.makedirs('./saved_obj')

if 'lemma_data.pkl' not in os.listdir('./saved_obj'):
    # mengcopy data yang telah diproses regex
    lemma_data = prepocessing_data.copy()
    # kemudian panggil fungsi untuk mengkonversi setiap data atau kalimat
    lemma_data['content'] = lemma_data['content'].apply(lambda x: convert_to_lemma(x))

    # kemudian simpan objek lemma_data
    with open('./saved_obj/lemma_data.pkl', 'wb') as f:
        pickle.dump(lemma_data, f)
else:
    # Load data jika file sudah ada atau dibuat
    with open('./saved_obj/lemma_data.pkl', 'rb') as f:
        lemma_data = pickle.load(f)


# proses mengcopy data final prepoccessing
final_data = lemma_data.copy()

# tokenisasi setiap kata menggunakan split()
final_data['content'] = final_data['content'].apply(lambda x: x.split())




# proses menggunakan class untuk proses indexing
if 'inverted_index.pkl' not in os.listdir('./saved_obj'):
    # Inisialisasi inverted objeknya
    inverted_index = index_builder(final_data['content'].values)

    #panggil setiap fungsi untnuk mendapatkan nilai frekuensi kata, frekuensi dokumen, dan indexnya
    inverted_index.count_terms_frequency()
    inverted_index.count_documents_frequency()
    inverted_index.construct_index()

    # proses menyimpan dalam bentuk objek ke pickle
    with open('./saved_obj/inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
else:
    # jika sudah diinisialisasikan untuk filenya
    inverted_index = None

    # load data yang telah diisimpan ke file pickle
    with open('./saved_obj/inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)

# Membuat model bm25 dari class 
model = BM25_OKAPI(inverted_index, df)

# simpan model ke pickle object untuk deploy
with open('./saved_obj/search_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# proses modifikasi
# buat rute agar tampilan langsung mengarah ke file index.html
@app.route('/')
def index():
    return render_template('index.html')

# proses modifikasi
# buat rute untuk proses searching, maka akan langsung menjalankan fungsi query_term
@app.route('/search')
def query_term():
    # ambil kata yang diinputkan pada form di variable 'q'
    query = request.args.get('q')
    # buat alternatif jika query kosong maka return berupa list kosong 
    if not query:
        return jsonify([])
    # prepocessing untuk memastikan query merupakan lower, bebas dari simbol dan stemming
    query = query.lower()
    query = ' '.join(re.findall(r'\w+', query))
    query = convert_to_lemma(query)

    # proses mencocokkan query dengan mmanggil fungsi 
    matched_index = model.match_query(query)
    # proses untuk mendapatkan hasil query setelah melakukan pencocokkan
    result = model.get_search_result(matched_index)

    # ubah data hasil query menjadi dictionary
    data = result.to_dict(orient='records')
    # retun data berupa json dan mengembalikkan respon http
    return jsonify(data)

# proses modifikasi
# proses menjalankan framework flusk dengan fungsi run
if __name__ == "__main__":
    app.run(debug=True)
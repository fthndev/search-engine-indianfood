import re
import os
import pickle
from collections import defaultdict
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('en_core_web_sm')

# proses modfikasi
# membuat dua variabel untuk menginisialisasikan stopword dan lemmatizingnya
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# membuat class indexnya
class index_builder:
    # buat koonstructor yang menerima satu parameter yang terdiri dari text, indexnya, total_documents, term_frequency, document_lengths,
    # dan average_document_length
    def __init__(self, text):
        self.text = text
        self.index = defaultdict(list)
        self.total_documents = len(text)
        self.terms_frequency = defaultdict(defaultdict)
        self.documents_frequency = defaultdict(int)
        self.documents_length = self.get_documents_length(text)
        self.average_documents_length = self.get_average_documents_length(text)

    # proses mendapatkan frekuensi kata dari setiap kata dalam dokumen
    def count_terms_frequency(self):
        # inisialisasi waktu mulai
        start_time = datetime.now()

        # gunakan perulangan untuk mengiterasikan setiap kata dalam dokumen
        for index, sentence in enumerate(self.text):
            for token in sentence:
                if(self.terms_frequency[token].get(index, -1) == -1):
                    # Inisialisasi jumlah kata jika belum ditemukan
                    self.terms_frequency[token][index] = 1
                else:
                    # tambahkan nilai frekuensi dengan angka 1 jika kata ditemukan kembali pada proses iterasi
                    self.terms_frequency[token][index] += 1

        #  tampilkan proses operasi
        print("Count terms frequency operation done in {} seconds".format((datetime.now()-start_time).seconds))

    # fungsi untuk mendapatkan nilai pada setiap dokumen untuk kata yang muncul
    def count_documents_frequency(self):
        # Inisialisasi waktu mulai
        start_time = datetime.now()

        # perulangan untuk mengiterasikan setiap kata
        for index, sentence in enumerate(self.text):
            for token in set(sentence):
                # tambahkan frekuensi dokumen dengan 1 jika kata dengan indeks token ditemukan dalam iterasi
                self.documents_frequency[token] += 1

        #  tampilkan proses operasi
        print("Count documents frequency operation done in {} seconds".format((datetime.now()-start_time).seconds))

    # buat fungsi index untuk setiap dokumen id dalam kata
    def construct_index(self):
        # Inisialisasi waktu mulai
        start_time = datetime.now()

        # perulangan untuk mengiterasikan setiap kata
        for index, sentence in enumerate(self.text):
          # perulangan untuk mengiterasikan setiap kata yang terurut dan tidak duplikat menggunakan set
            for token in set(sentence):
                # tambahkan id indeksnya ke inverted index dengan mengakses key berupa kata
                self.index[token].append(index)

        #  tampilkan proses operasi
        print("Construct inverted index operation done in {} seconds".format((datetime.now()-start_time).seconds))

    # proses menghitung panjang setiap dokumen
    def get_documents_length(self, docs):
        # membuat dictionary dengan nilai default 0
        doc_len_dict = defaultdict(int)

        for index, doc in enumerate(docs):
            #proses menghitung panjang dari kalimat dengan menggabungkan setiap kata menggunakan join
            doc_len_dict[index] = len(' '.join(doc))

        return doc_len_dict

    # buat fungsi untuk mendapatkan nilai rata-rata dengan berdasarkan total karakternya dalam kalimat dibagi dengan total kalimat
    def get_average_documents_length(self, docs):
        return sum(len(' '.join(doc)) for doc in docs)/len(docs)

# proses Modifikasi
# konversi ke lemma
def convert_to_lemma(x):
    # mengubah kata atau term dengan format lower dan terbebas dari simbol
    text = x.lower()
    text = ' '.join(re.findall(r'\w+', text))
    # membuat objek varibel x menggunakan modul spaCy dengan memproses stemming menggunakan lemmazation
    sentence = nlp(text)
    # proses menampung data yang sudah terbebas dari stopwords, bukan tanda baca, dan spasi
    lemma = [token.lemma_ for token in sentence if token.text not in stop_words and not token.is_punct and not token.is_space]

    # gabungkan kata yang telah di konversi
    return ' '.join(lemma)  
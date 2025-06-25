from collections import defaultdict
from datetime import datetime
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import log10
import spacy

nlp = spacy.load('en_core_web_sm')

# proses modifikasi
# membuat dua variabel untuk menginisialisasikan stopword dan lemmatizingnya
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# class bm25 okapi
class BM25_OKAPI:
    # buat constructor dengan dua parameter
    def __init__(self, inv_index, docs):
        start_time = datetime.now()
        self.inv_index = inv_index
        self.docs = docs
        self.tf = self.generate_tf(inv_index.terms_frequency)
        self.idf = self.generate_idf(inv_index.documents_frequency)
        print('BM25 OKAPI model succesfully generated - {} seconds'.format((datetime.now()-start_time).seconds))

    # koneversi query menjadi indeks yang cocok
    def process_query(self, base_query):
        # proses mengubah query menjadi lowecase
        base_query = base_query.lower()
        # proses menghilangkan simbol atau emoticon
        base_query = ' '.join(re.findall('\w+', base_query))

        # proses modifikasi
        # menggunakan spacy untuk proses stemming 
        doc = nlp(base_query)
        # menyimpan data dalam bentuk list
        lemmatized_tokens = [token.lemma_ for token in doc]
        # menyimpan data yang telah diproses menggunakan lemma dan bukan stopword
        query = [token.lemma_ for token in doc if token.text not in stop_words]

        # proses tokenisasi kalimat
        base_query = base_query.split()

        # Inisialisasi variabel dengan data collection set agar unik
        stopword_index = set()
        token_index= set()
        query_word_frequency = defaultdict(int)

        for token in lemmatized_tokens:
            # proses mengkonversi setiap kata menjadi lemma
            # proses menyimpan setiap freskuensi kata
            query_word_frequency[token] += 1

            # proses menyimpan kata stopword dan kata yang bukan stopword ke dalam masing masing variabel yang telah diset
            if token in stop_words:
                stopword_index |= set(self.inv_index.index.get(token, []))
            else:
                token_index |= set(self.inv_index.index.get(token, []))

        if(len(token_index) == 0):
            # proses menambahkan stopword ke variabel token index jika jumlah karakternya 0
            token_index |= stopword_index
        elif((len(token_index) + len(stopword_index)) == 0):
            # proses mengeset nilai jik masing-masing panjang nilainya 0, set dengan string the
            token_index |= set(self.inv_index.index.get('the', []))

        # proses mengkonversi setiap kata pada query menjadi lemma

        return query, token_index, query_word_frequency

    # proses menghitug frekuensi kata berdasarkan bm25
    def generate_tf(self, terms_frequency_):
        # inisialisasi nilai untuk kontrol pengaruh panjang dokumen dalam menentukan skor
        b = 0.75
        # nilai untuk mengontrol nilai tf
        k = 5

        # inisialisasi data berupa dictionary dengan nilai 0
        tf_result = defaultdict(defaultdict)

        # gunakan perulangan berdasarkan keys dalam dictionary
        for term in terms_frequency_.keys():
            for doc_id in terms_frequency_[term].keys():
                # proses mendapatkan panjang dokumen
                doc_len = self.inv_index.documents_length[doc_id]
                # proses mendapatkan nilai rata-rata dokumen
                avg_doc_len = self.inv_index.average_documents_length

                # proses menghitung nilai dengan cara nilai kontrol ditambah 1 kemudian dikalikan dengan jumlah frekuensi kata
                numerator = (k+1) * terms_frequency_[term][doc_id]
                denominator = terms_frequency_[term][doc_id] + k*(1-b+(b*doc_len/avg_doc_len))
                # proses menyimpan nilai tf
                tf_result[term][doc_id] = numerator/denominator

        return tf_result

    # proses menghitung nilai idfnya dengan membuat fungsi
    def generate_idf(self, documents_frequency_):
        # inisialisasi dictionary untuk menyimpan nilai dengan nilai awal adalah 0
        idf_result = defaultdict(int)

        # proses mengiterasikan berdasarkan keys
        for term in documents_frequency_.keys():
            # proses menghitung idfnya dengan cara total dokumen dikurangi dengan frekuensi dokumen ditambah 0.1 dan dibagi dengan
            # dokumen frekuensi + 0.1. setelah itu dihitung nilai log
            idf_result[term] = log10((self.inv_index.total_documents - documents_frequency_[term] + 0.1)/
                                     (documents_frequency_[term]+0.1))

        return idf_result

    # proses pencocokan query
    def match_query(self, base_query):
        # inisialisasi waktu awal proses
        start_time = datetime.now()
        # buat tiga variabel untuk menampung nilai hasil return dari fungsi process_query
        terms, found_index, query_word_frequency = self.process_query(base_query)

        # Inisialisasi nilai dictionary dengan 0
        match_score = defaultdict(int)

        # gunakan perulangan pada kata
        for index in found_index:
            score = 0
            for term in terms:
                # proses untuk mendapatkan nilai tf, idf, dan frekuensi setiap kata
                tf_val = self.tf.get(term).get(index)
                idf_val = self.idf.get(term)
                qw_freq = query_word_frequency.get(term)

                # proses pengecekan jika kata tidak ditemukan pada tf, idf, frekuensi kata dan gunakan continue untuk meleewati proses
                if((tf_val==None) or (idf_val==None) or (qw_freq==None)):
                    continue
                else:
                    # proses mendapatkan skor dengan cara nilai score di increamennt dengan hasil perkalian tf, idf, dan frekuensi kata
                    score = score + (qw_freq * tf_val * idf_val)

            # proses set nilai ke dictionary
            match_score[index] = score

        # kemudian terdapat proses untuk mengeset score yang dibawah quantile 0.6
        threshold_score = np.quantile(list(match_score.values()), .60)

        # proses untuk hasil pencocokan terbauk dengan descending scorenya
        sorted_match_score = sorted(match_score.items(), key=lambda x: x[1], reverse=True)

        # proses mengeleminasi dokumen yang nilai scorenya dibawah treshold quantile
        sorted_match_score = [(idx, score) for idx, score in sorted_match_score if score >= threshold_score]

        # manampilkan operasi
        finish_time = (datetime.now() - start_time).microseconds
        print("{} News found in {} s".format(len(sorted_match_score), finish_time/1000000))

        return dict(sorted_match_score)

    # proses modifikasi
    # fungsi untuk mendapatkan hasil search
    def get_search_result(self, index):
        # proses mengambil data berdasarkan dengan indeks key
        indeks = list(index.keys())
        # proses mendapatkan data berdasarkan dengan valuenya
        score = list(index.values())

        # menyalin data berdasarkan keynya 
        result_df = self.docs.iloc[indeks].copy()
        # menambahkan data dengan key score dan valuenya berupa score hasil perhitungan bm25
        result_df['score'] = score
        # peroses mengembalikan nilai
        return result_df
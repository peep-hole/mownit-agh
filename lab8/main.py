import math
from tkinter import *
import nltk
import scipy.sparse
import numpy as np
import sklearn
import sklearn.decomposition
import matplotlib.pyplot as plt


MINIMAL_CORR = 0.2
SVD_K = 1000


def get_words_and_articles(file):
    f = open(file)
    for article in f.read().split('\n\n'):
        name, content = article.split('\n', 1)
        yield name, content


def query_to_vector(q, terms, term_to_ind, tokens):

    term_set = set()
    vec = scipy.sparse.lil_matrix((len(terms), 1))
    for token in tokenizer(q):
        comparable = stemmer.stem(token)
        if comparable in tokens:
            term_set.add(comparable)

    if len(term_set) == 0:
        return None
    value = 1.0 / math.sqrt(len(term_set))
    for tok in term_set:
        vec[term_to_ind[tok], 0] = value

    return vec


def plot_corr_vector(v):
    x = [i for i in range(len(v))]
    y = [el[1] for el in v]
    y2 = [MINIMAL_CORR for xi in x]
    plt.plot(x, y, 'bo', x, y2, 'r-')
    plt.show()


tokenizer = nltk.word_tokenize
stemmer = nltk.stem.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')


def save_as_svd(matrix):
    svd = sklearn.decomposition.TruncatedSVD(SVD_K).fit(matrix.T)
    res_matrix = svd.transform(matrix.T)
    comps = svd.components_

    return res_matrix, comps


def save_matrix():
    art_dest = './articles/corpus.txt'

    w = get_words_and_articles(art_dest)
    w2 = get_words_and_articles(art_dest)

    tokens = set()
    articles = dict()

    percentage = 0

    print("TOKENIZING...")
    a_count = len(list(w2))
    for i, word in enumerate(w):
        articles[i] = word
        if (i / a_count) * 100 > percentage:
            print("PROGRESS: ", percentage, "%")
            percentage += 5
        for token in tokenizer(word[1]):
            token = stemmer.stem(token)
            if len(token) > 2 and token.isalnum() and not token.isdigit() and token not in stop_words:
                tokens.add(token)

    terms = dict()
    term_to_ind = dict()
    for i, token in enumerate(tokens):
        terms[i] = token
        term_to_ind[token] = i

    term_by_doc = scipy.sparse.lil_matrix((len(terms), len(articles)))

    percentage = 0
    a_count = len(articles)
    print("COUNTING...")
    for i in range(len(articles)):
        if (i / a_count) * 100 > percentage:
            print("PROGRESS: ", percentage, "%")
            percentage += 5
        for token in tokenizer(articles[i][1]):
            ter = stemmer.stem(token)
            if ter in term_to_ind:
                term_by_doc[term_to_ind[ter], i] += 1

    count = scipy.sparse.linalg.norm(term_by_doc, ord=0, axis=1)
    idf = np.log(float(len(articles)) * np.reciprocal(count, dtype=np.float64))

    term_by_doc = term_by_doc.T.multiply(idf).T

    term_by_doc = sklearn.preprocessing.normalize(term_by_doc, axis=0, norm='l2', copy=False)

    return term_by_doc, terms, articles, term_to_ind, tokens


def search_engine(is_svd=False):
    term_by_doc, terms, articles, term_to_ind, tokens = save_matrix()

    if is_svd:
        matrix, comp = save_as_svd(term_by_doc)

    while True:
        query = input("SEARCH: ")
        q_vec = query_to_vector(query, terms, term_to_ind, tokens)
        if q_vec is None:
            print("")
            print("Could not find such thing")
            print("")
            continue

        if is_svd:
            q_form = comp.dot(q_vec.todense())
            cor_svd = matrix.dot(q_form)
            corr = [((0, ind), cor_svd[ind, 0]) for ind in range(len(articles))]
        else:
            corr = list(q_vec.T.dot(term_by_doc).todok().items())
        corr.sort(key=lambda x: -x[1])
        for el in corr:
            if el[1] > MINIMAL_CORR:
                print("-------------------------------------------------------------------")
                print(" ")
                print(articles[el[0][1]][0])
                print(articles[el[0][1]][1])
                print(" ")
            else:
                break
        plot_corr_vector(corr)


def search_query(q, is_svd=False, matrix_tools=None, svd_tools=None):
    if matrix_tools is None:
        term_by_doc, terms, articles, term_to_ind, tokens = save_matrix()
    else:
        term_by_doc, terms, articles, term_to_ind, tokens = matrix_tools

    if is_svd and svd_tools is None:
        svd_tools = save_as_svd(term_by_doc)
    matrix, comp = svd_tools

    query = q
    q_vec = query_to_vector(query, terms, term_to_ind, tokens)
    if q_vec is None:
        print("")
        print("Could not find such thing")
        print("")
        return None

    if is_svd:
        q_form = comp.dot(q_vec.todense())
        cor_svd = matrix.dot(q_form)
        corr = [((0, ind), cor_svd[ind, 0]) for ind in range(len(articles))]
    else:
        corr = list(q_vec.T.dot(term_by_doc).todok().items())
    corr.sort(key=lambda x: -x[1])
    res_list = []
    for el in corr:
        if el[1] > MINIMAL_CORR:
            res_list.append((articles[el[0][1]][0], articles[el[0][1]][1]))
        else:
            break
    plot_corr_vector(corr)
    return res_list


def tk(is_SVD=False, matrix_tools=None, svd_tools=None):
    root = Tk()
    root.geometry("1000x1000")
    root.title(" Moownit ")

    def Take_input():
        INPUT = inputtxt.get("1.0", "end-1c")
        res = search_query(INPUT, is_svd=is_SVD, matrix_tools=matrix_tools, svd_tools=svd_tools)
        if res is not None:
            for el in res:
                Output.insert(END, "\n")
                Output.insert(END, el[0])
                Output.insert(END, "\n")
                Output.insert(END, el[1])
                Output.insert(END, "\n")
        else:
            Output.insert(END, "Wrong query")

    l = Label(text="Search")
    inputtxt = Text(root, height=5,
                    width=100,
                    bg="light yellow")

    Output = Text(root, height=800,
                  width=800,
                  bg="light cyan")

    Display = Button(root, height=2,
                     width=20,
                     text="Search",
                     command=lambda: Take_input())

    l.pack()
    inputtxt.pack()
    Display.pack()
    Output.pack()

    mainloop()

def main():
    matrix_tools = save_matrix()
    svd_tools = save_as_svd(matrix_tools[0])
    while True:
        tk(matrix_tools=matrix_tools, is_SVD=True, svd_tools=svd_tools)


main()




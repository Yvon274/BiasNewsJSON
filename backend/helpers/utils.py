import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QueryChecker:
    n_feats = 5000

    def __init__(self, query = None, docs = None):
        self.tfidf_vec = None
        self.article_name_to_url = None
        self.data = None
        self.numArticles = None
        self.article_url_to_index = None
        self.article_id_to_name = None
        self.query = query
        self.docs = docs
        self.topArticles = []

    def build_vectorizer(self, max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
        """Sets a TfidfVectorizer object with the above preprocessing properties.

        Note: This function may log a deprecation warning. This is normal, and you
        can simply ignore it.

        Parameters
        ----------
        max_features : int
            Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer
            constructer.
        stop_words : str
            Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer.
        max_df : float
            Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer.
        min_df : float
            Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer.
        norm : str
            Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer.

        Returns
        -------
        TfidfVectorizer
            A TfidfVectorizer object with the given parameters as its preprocessing properties.
        """
        return TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df,
                               norm=norm)
        
    def loadData(self, path):
        """
        path : path to the data on local machine
        """
        with open(path) as f:
            data = json.load(f)['articles']
            # data = json.loads(data.readlines()[0])
            # data = json.loads(['articles'].readlines()[0])
        self.data = data
        self.numArticles = len(data)

        self.article_url_to_index = {article_url: index for index, article_url in enumerate([d['url'] for d in data])}

        self.article_name_to_url = {name: mid for name, mid in zip([d['title'] for d in data],
                                                           [d['url'] for d in data])}
        self.article_url_to_name = {v: k for k, v in self.article_name_to_url.items()}
        
    
    def get_most_similar(self, query, data):
        """Returns a float giving the cosine similarity of
           the two movie transcripts.
        
        Params: {query: query in string form.
                 mov2 (str): Name of the article.
                 input_doc_mat (numpy.ndarray): Term-document matrix of articles, where
                        each row represents a document (movie transcript) and each column represents a term.
                 movie_name_to_index (dict): Dictionary that maps movie names to the corresponding row index
                        in the term-document matrix.}
        Returns: Float (Cosine similarity of the two movie transcripts.)
        """
        self.data = data

        vectorizer = self.build_vectorizer(QueryChecker.n_feats, 'english')
        corpus = self.data['text'].str.lower()


        tfidf_matrix = vectorizer.fit_transform(corpus)

        query_tfidf = vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        most_similar_index = np.argmax(cosine_similarities)
        most_similar_document = corpus[most_similar_index]
        similarity_score = cosine_similarities[most_similar_index]

        top_50_indices = np.argsort(cosine_similarities)[-50:][::-1]
        return top_50_indices
        

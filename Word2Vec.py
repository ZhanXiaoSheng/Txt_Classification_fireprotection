from gensim.models import Word2Vec, word2vec

#sentences = word2vec.Text8Corpus('C:\\Users\\zhangsheng\\Desktop\\DSjjcl\\文本分类（报警信息）\\new_data2.txt')
#model = Word2Vec(sentences, size=200, min_count=2)
#model.save('word2vec.model')

#print(model.similarity('积水', '交通'))

sentences = word2vec.Text8Corpus('K:\\data\\cut.txt')
model = Word2Vec(sentences,size=200,min_count=5)
model.save('cut.model')

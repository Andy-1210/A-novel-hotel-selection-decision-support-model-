import pandas as pd
from pyltp  import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
ldir='D:\\xinjianwenjianjia (4)\\ltp_data_v3.4.0\\ltp_data_v3.4.0\\cws.model'  #分词模型
dicdir='word'                           #外部字典

raw = pd.read_csv('D:/桌面/123.txt', names=['txt'], sep='aaa', encoding='utf-8',engine='python')

segmentor = Segmentor()                             #初始化实例
segmentor.load_with_lexicon(ldir, 'word')    #加载模型

raw['cut'] = raw.txt.apply(segmentor.segment)

from gensim.models.word2vec import Word2Vec
w2vmodel = Word2Vec(size=300, min_count=10)
w2vmodel.build_vocab(raw.cut)
w2vmodel.train(raw.cut, total_examples=w2vmodel.corpus_count, epochs=10)#可调参数优化模型
print(w2vmodel.wv.most_similar('服务', topn=200))
#w2vmodel.save('word2vec.model')



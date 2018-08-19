import gensim
import logging

class MyClass:


    def set_up_logging(self):
        # setup logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def load_model(self):
        print('---- loading model - takes a few minutes')
        self.logger.info('---- loading model - takes a few minutes')
        self.model = gensim.models.KeyedVectors.load_word2vec_format('/temp/kaggle/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

    def run_example(self):
        x = self.model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
        print(x)
        print('----')
        x = self.model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
        print(x)
        print('----')
        x = self.model.wv.doesnt_match("breakfast cereal dinner lunch".split())
        print(x)
        print('----')
        x = self.model.wv.similarity('woman', 'man')
        print(x)
        print('----')


    def print_similar(self, word):
        print(word)
        print(self.model.wv.most_similar(positive=[word],topn=100))

    def generate_karpman_drama_triangle_terms(self):
        logging.info('hi')

        print('_____________')
        self.print_similar('victim')
        self.print_similar('sufferer')
        self.print_similar('casualty')
        self.print_similar('loser')

        print('_____________')
        self.print_similar('persecutor')
        self.print_similar('winner')
        self.print_similar('guard')
        self.print_similar('marshal')
        self.print_similar('bully')

        print('_____________')
        self.print_similar('rescuer')
        self.print_similar('helper')
        self.print_similar('caretaker')



if __name__ == '__main__':
    c = MyClass()
    c.set_up_logging()
    c.load_model()
    c.run_example()
    c.generate_karpman_drama_triangle_terms()
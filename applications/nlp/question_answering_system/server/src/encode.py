from towhee import ops, pipe


class SentenceModel:
    """
    SentenceModel
    """

    def __init__(self):
        self.sentence_embedding_pipe = (
            pipe.input('sentence')
                .map('sentence', 'embedding', ops.sentence_embedding.sbert(model_name='all-MiniLM-L12-v2'))
                .map('embedding', 'embedding', ops.towhee.np_normalize())
                .output('embedding')
            )

    def sentence_encode(self, data_list):
        res_list = []
        for data in data_list:
            res = self.sentence_embedding_pipe(data)
            res_list.append(res.get()[0])
        return res_list


if __name__ == '__main__':
    MODEL = SentenceModel()
    # Warm up the model to build image
    MODEL.sentence_encode(['hello world'])

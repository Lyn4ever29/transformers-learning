import os

class CommonConfig:
    bert_dir = "/data1/model/chinese-macbert-base"
    output_dir = "./checkpoint/"
    data_dir = "/data1/Bert-BiLLSTM-CRF/origin"


class NerConfig:
    def __init__(self):
        cf = CommonConfig()
        self.bert_dir = cf.bert_dir
        self.output_dir = cf.output_dir
        # self.output_dir = os.path.join(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = cf.data_dir

        labels_list = ['O', 'B-HCCX', 'I-HCCX', 'B-HPPX', 'I-HPPX', 'B-XH', 'I-XH', 'B-MISC', 'I-MISC']
        self.num_labels = len(labels_list)
        self.label2id = {l: i for i, l in enumerate(labels_list)}
        self.id2label = {i: l for i, l in enumerate(labels_list)}

        self.max_seq_len = 512
        self.epochs = 3
        self.train_batch_size = 64
        self.dev_batch_size = 64
        self.bert_learning_rate = 3e-5
        self.crf_learning_rate = 3e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500

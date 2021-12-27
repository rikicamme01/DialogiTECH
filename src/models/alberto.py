from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

class AlBERTo(BertForSequenceClassification):
    def __init(self):
        super(AutoConfig.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'))
        


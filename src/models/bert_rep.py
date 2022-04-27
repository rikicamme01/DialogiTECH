from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch 

class BertRep():
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML')
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/RepML')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text:list[str]):
        text_tensor = self.tokenizer.encode(text,
                                    return_special_tokens_mask=True,
                                    return_offsets_mapping=True,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    ).to(self.device)

        output = self.model(text_tensor)
        return output 

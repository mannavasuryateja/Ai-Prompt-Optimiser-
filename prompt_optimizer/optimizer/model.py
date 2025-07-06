from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return model, tokenizer

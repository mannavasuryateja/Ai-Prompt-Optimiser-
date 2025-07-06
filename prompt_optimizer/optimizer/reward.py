from optimizer.utils import call_llm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

model_name = 'all-MiniLM-L6-v2'
reward_model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def reward_function(original_prompt, optimized_prompt):
    original_output = call_llm(original_prompt)
    optimized_output = call_llm(optimized_prompt)
    similarity = util.cos_sim(
        reward_model.encode(original_output, convert_to_tensor=True),
        reward_model.encode(optimized_output, convert_to_tensor=True)
    )[0].item()
    token_penalty = len(tokenizer.tokenize(optimized_prompt)) * 0.01
    return similarity - token_penalty

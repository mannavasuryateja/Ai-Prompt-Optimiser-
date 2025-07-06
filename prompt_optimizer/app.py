import gradio as gr
from optimizer.model import load_model_and_tokenizer
from optimizer.reward import reward_function
from optimizer.utils import call_llm

model, tokenizer = load_model_and_tokenizer()

def optimize_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=32)
    optimized = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = call_llm(optimized)
    reward = reward_function(prompt, optimized)
    return optimized, response, reward

demo = gr.Interface(
    fn=optimize_prompt,
    inputs="text",
    outputs=["text", "text", "number"],
    title="AI Prompt Optimizer",
    description="Shortens prompts while maintaining or improving output quality."
)

demo.launch()

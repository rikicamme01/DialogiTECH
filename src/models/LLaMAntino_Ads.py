#%%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "swap-uniba/LLaMAntino-2-7b-hf-ITA"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)


prompt = "Cosa posso fare oggi pomeriggio?"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids=input_ids)

print(tokenizer.batch_decode(outputs.detach().cpu().numpy()[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model + tokenizer globally for performance
MODEL_PATH = "EleutherAI/gpt-neo-1.3B"  # Replace with GPT-Neo model path

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


def run_query_pipeline(nl_query: str) -> str:
    """
    Given a natural language question, generate an SQL query using GPT-Neo.
    """
    prompt = f"Translate the following question to SQL:\nQuestion: {nl_query}\nSQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if the result contains valid SQL
        if not result or "SQL:" not in result:
            return "Error: Model output does not contain a valid SQL query."

        # Extract SQL from the model output
        sql_query = result.split("SQL:")[-1].strip()
        return sql_query

    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

from moe_reft import sft_dataset
from transformers import AutoTokenizer

tokenizer_model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
response_template = sft_dataset.extract_response_template(tokenizer)

train_dataset = sft_dataset.SFTDataset(
    source="openai/gsm8k",
    tokenizer=tokenizer,
    response_template_ids=tokenizer(response_template)["input_ids"],
    system_key=None,
    system_message="You are a helpful math tutor. Solve step by step.",
    user_key="question",
    assistant_key="answer",
    split="train",
    name="main",
)

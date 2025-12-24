#!/usr/bin/env python3
"""
Script to modify the notebook for Mac compatibility and complete all TODOs.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("starter/gen_ai_fundamentals_project_starter.ipynb")

# Cell 3: Replace nvidia-smi with MPS check
CELL_3_NEW = '''# Verify we have MPS (Metal Performance Shaders) available on Apple Silicon
# For Mac compatibility - replaces nvidia-smi check
# No changes needed in this cell

import torch
import platform

print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\\n[OK] Using MPS device for Apple Silicon GPU acceleration")
    # Test MPS with a small tensor operation
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    print(f"MPS test successful: tensor shape {y.shape}")
else:
    device = torch.device("cpu")
    print(f"\\n[INFO] MPS not available, using CPU")
'''

# Cell 4: Replace Unsloth with standard PEFT for Mac compatibility
CELL_4_NEW = '''# Load the Qwen 2.5 3B Instruct model with standard PEFT for Mac compatibility
# This replaces Unsloth which requires CUDA
# TODO: Completed - LoRA rank and target modules configured

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

max_seq_length = 384  # Increase if you get errors about the sequence length

# Set the LoRA rank - choosing 64 as it provides a good balance between
# model capacity and training efficiency. Higher ranks (128) give more
# capacity but slower training; lower ranks (8,16) are faster but less expressive.
# For this letter-counting task, 64 provides sufficient capacity.
# Valid options: {8, 16, 32, 64, 128}
lora_rank = 64

# Device selection for Apple Silicon Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float16  # MPS works best with float16
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    print("Using CUDA")
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    print("Using CPU")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Load the model - NO quantization on Mac (bitsandbytes not supported)
# We use float16 for memory efficiency on Apple Silicon
print("Loading model... this may take a few minutes...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch_dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    attn_implementation="eager",  # MPS doesn't support flash attention
)

# Move model to MPS if applicable
if device.type == "mps":
    model = model.to(device)

# Configure LoRA adapters
# Target modules explanation:
# - q_proj, k_proj, v_proj, o_proj: Attention projection layers (core for reasoning)
# - gate_proj, up_proj, down_proj: MLP layers (help with output generation)
# Including all these modules provides comprehensive fine-tuning coverage
# while maintaining parameter efficiency.
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank,  # Setting alpha=rank is a common practice
    target_modules=[
        "q_proj",    # Query projection - attention mechanism
        "k_proj",    # Key projection - attention mechanism
        "v_proj",    # Value projection - attention mechanism
        "o_proj",    # Output projection - attention output
        "gate_proj", # MLP gate projection
        "up_proj",   # MLP up projection
        "down_proj", # MLP down projection
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

print(f"\\nModel loaded successfully on {device}")
print(f"LoRA rank: {lora_rank}")
print(f"Target modules: {lora_config.target_modules}")

# Helper function for text generation (replaces vLLM)
def generate_completion(model, tokenizer, text, max_new_tokens=2048, temperature=0.8, top_p=0.95):
    """Generate text completion using standard HuggingFace generate method."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion (exclude the prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return generated_text
'''

# Cell 6: Update to use generate_completion instead of vLLM
CELL_6_NEW = '''# First, let's see what happens when we have a blank system prompt
# No changes needed in this cell
SYSTEM_PROMPT = """"""
USER_PROMPT = 'How many of the letter "g" are there in the word "engage"'

# Convert the chat messages to a single string so the model can complete it
text_for_completion = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT,
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# Generate using standard HuggingFace method (Mac compatible, replaces vLLM)
output = generate_completion(
    model, tokenizer, text_for_completion,
    max_new_tokens=2048, temperature=0.8, top_p=0.95
)

# Print the text input for the model and the model's output
print("=== TEXT FOR COMPLETION ===")
print(text_for_completion)
print("=== GENERATED OUTPUT ===")
print(output)
'''

# Cell 8: SYSTEM_PROMPT with CoT and few-shot example
CELL_8_NEW = '''# Let's work on a new system prompt that will help the model break this problem
# down into steps, for example, using "letter-by-letter" spelling.
# TODO: Completed - CoT prompt with few-shot example

# Chain-of-Thought prompt with at least one few-shot example
# This helps the model understand the step-by-step reasoning process
SYSTEM_PROMPT = """You are a helpful assistant that counts letter occurrences in words.
You must follow a precise step-by-step reasoning process.

For each question, you will:
1. Spell out the word letter by letter
2. For each letter, check if it matches the target letter
3. Keep a running count of matches
4. Provide your final answer in a specific format

Here is an example:

Question: How many of the letter "o" are there in the word "room"?
<reasoning>
Counting the number of o's in the word room
1. r - 0 o's so far
2. o - 1 o's so far
3. o - 2 o's so far
4. m - 2 o's so far

The letter "o" appears 2 times in the word "room".
</reasoning>
<answer>
2
</answer>

Now solve the user's question using the same format. Always use numbered steps and track your count after each letter."""


USER_PROMPT = 'How many of the letter "g" are there in the word "engage"'

# Convert the chat messages to a single string so the model can complete it
text_for_completion = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT,
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# Generate using standard HuggingFace method (Mac compatible)
output = generate_completion(
    model, tokenizer, text_for_completion,
    max_new_tokens=2048, temperature=0.8, top_p=0.95
)

# Print the text input for the model and the model's output
print("=== TEXT FOR COMPLETION ===")
print(text_for_completion)
print("=== GENERATED OUTPUT ===")
print(output)
'''

# Cell 14: Update to use generate_completion
CELL_14_NEW = '''# Let's see how well the model runs out-of-the-box
# No changes needed in this cell

text = tokenizer.apply_chat_template(
    ds[0]["prompt"], tokenize=False, add_generation_prompt=True
)

# Generate using standard HuggingFace method (Mac compatible)
output = generate_completion(
    model, tokenizer, text,
    max_new_tokens=1024, temperature=0.8, top_p=0.95
)

print(output)
'''

# Cell 17: numbering_reward_func TODO completed
CELL_17_NEW = '''# Let's work on a function that rewards correct numbering in the bullet points
# When using GRPO, we lean on reward functions that are relatively easy to
# compute, thus removing the need to have a second large model just for
# evaluation.
# In this case, we'll use regular expressions quite a bit.
# TODO: Completed - reward values filled in


def extract_letter_numbering(response):
    """Extract the numbers at the beginning of the line

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    4. a - 2 so far
    5. l - 2 so far
    returns [1, 2, 3, 4, 5]
    """
    import re

    # We use a regular expression to find lines of the form:
    # '\\n[number]. [letter]'
    pattern = r"\\n(\\d+). [a-z]"

    # Use `re` to find all matches of the pattern in the response
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if matches:
        return [int(m) for m in matches]
    return []


assert extract_letter_numbering(
    """
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. a - 2 so far
5. l - 2 so far
"""
) == [1, 2, 3, 4, 5]


def numbering_reward_func(completions, words, **kwargs) -> list[float]:
    """Provides a reward for getting the numbering at the beginning of the line correct

    1. g - 1 so far <-- Good in-order numbering
    2. o - 1 so far <-- Good in-order numbering
    3. a - 2 so far <-- Good in-order numbering
    3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4
    1. l - 2 so far <-- Bad numbering, extra letter and out-of-order
    1. l - 2 so far <-- Bad numbering, extra letter and out-of-order

    """
    responses = [completion[0]["content"] for completion in completions]

    res = []
    for response, word in zip(responses, words):
        reward = 0

        for ix, spell_number in enumerate(extract_letter_numbering(response)):
            line_number = ix + 1

            # Get points for in-order numbering
            if spell_number == line_number:
                # Reward for correct sequential numbering (+0.5)
                reward += 0.5
            # Otherwise lose points
            else:
                # Penalty for out-of-order numbering (-0.5)
                reward -= 0.5

            # Lose extra points for continuing beyond the length of the word
            if line_number > len(word):  # We use the index of the line
                # Penalty for extra lines beyond word length (-1.0)
                reward -= 1.0

        res.append(reward / len(word))
    return res


res = numbering_reward_func(
    completions=[
        [
            {  # Worse response
                "content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far <-- Good in-order numbering
2. o - 1 so far <-- Good in-order numbering
3. a - 2 so far <-- Good in-order numbering
3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4
1. l - 2 so far <-- Bad numbering, extra letter and out-of-order
1. l - 2 so far <-- Bad numbering, extra letter and out-of-order
</reasoning>
<answer>2</answer>"""
            },
        ],
        [
            {  # Better response
                "content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far <-- Good in-order numbering
2. o - 1 so far <-- Good in-order numbering
3. a - 2 so far <-- Good in-order numbering
3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4
</reasoning>
<answer>2</answer>"""
            },
        ],
    ],
    words=["goal", "goal"],
)
print(res)

assert res[1] > res[0], "The better response should have a higher reward"
'''

# Cell 19: spelling_reward_func TODO completed
CELL_19_NEW = '''# Reward correct spelling of the word
# TODO: Completed - reward values filled in


def extract_spelling(response):
    """Extract the spelling from the response

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    3. l - 2 so far
    5. l - 2 so far
    Returns "goall"
    """
    import re

    pattern = r"\\n\\d+. ([a-z])"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if matches:
        return "".join([m for m in matches])
    return ""


assert extract_spelling(
    """Here is a letter by letter spelling:

1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
3. l - 2 so far
5. l - 2 so far
"""
) == "goall"


def spelling_reward_func(completions, words, **kwargs) -> list[float]:
    """A spelling reward function."""
    from collections import Counter

    responses = [completion[0]["content"] for completion in completions]

    res = []

    for word, response in zip(words, responses):
        reward = 0.0
        spelled = extract_spelling(response)

        # Provide a reward for exactly correct spelling (+2.0)
        if spelled == word:
            reward += 2.0

        # Provide a penalty for each letter of difference in length (-0.5 per letter)
        reward -= 0.5 * abs(len(spelled) - len(word))

        # Count letters in both
        spelled_counter = Counter(spelled.lower())
        word_counter = Counter(word.lower())

        # Provide a penalty for each letter that is in spelled but not in word (extra letters, -1.0 per letter)
        for letter, count in spelled_counter.items():
            if letter not in word_counter:
                reward -= 1.0 * count
            elif count > word_counter[letter]:
                reward -= 1.0 * (count - word_counter[letter])

        # Provide a penalty for each letter that is in word but missing from spelled (-0.5 per letter)
        for letter, count in word_counter.items():
            if letter not in spelled_counter:
                reward -= 0.5 * count
            elif spelled_counter[letter] < count:
                reward -= 0.5 * (count - spelled_counter[letter])

        res.append(reward)
    return res


res = spelling_reward_func(
    completions=[
        [  # Worse response
            {
                "content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. l - 2 so far
5. l - 2 so far
</reasoning>
<answer>2</answer>"""
            }
        ],
        [  # Better Response
            {
                "content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. l - 2 so far
</reasoning>
<answer>2</answer>"""
            }
        ],
    ],
    words=["goal", "goal"],
)

print(res)

assert res[1] > res[0], "The better response should have a higher reward"
'''

# Cell 21: counting_reward_func TODO completed
CELL_21_NEW = '''# Let's reward the model for properly counting the occurrences of a letter in a word
# TODO: Completed - condition and normalization filled in


def get_resp_letters_and_counts(response):
    """Extract the letters and counts from the response

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    4. a - 2 so far
    5. l - 2 so far
    returns [('g', 1), ('o', 1), ('a', 2), ('a', 2), ('l', 2)]
    """
    import re

    pattern = r"\\n(\\d+)\\. ([a-z])\\D*(\\d+)"

    # Find strings matching e.g. "2. a - 2 so far"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)

    if not matches:
        return []

    return [
        (matched_letter, matched_count_so_far)
        for _, matched_letter, matched_count_so_far in matches
    ]


assert get_resp_letters_and_counts(
    """
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. a - 2 so far
5. l - 2 so far
"""
) == [("g", "1"), ("o", "1"), ("a", "2"), ("a", "2"), ("l", "2")]


def counting_reward_func(completions, letters, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]

    res = []

    # Iterate over each of the letter-response pairs
    for letter, response in zip(letters, responses):
        reward = 0

        letters_and_counts = get_resp_letters_and_counts(response)

        # If there are no matches, provide a negative reward
        if not letters_and_counts:
            res.append(-1)
            continue

        # Start counting the matching letters
        actual_count = 0
        for resp_letter, resp_count in letters_and_counts:
            # If there's a match, count the letter
            if letter.lower() == resp_letter.lower():
                actual_count += 1

            # If the count is accurate, add a reward (+1.0), else subtract a reward (-1.0)
            if int(resp_count) == actual_count:
                reward += 1.0
            else:
                reward -= 1.0

        # Return the reward normalized by the length of the matches
        res.append(reward / len(letters_and_counts))
    return res


res = counting_reward_func(
    completions=[
        [  # Worse response
            {
                "content": """<reasoning>\\nHere is a letter by letter spelling:

1. g - 0 so far
2. o - 0 so far
3. a - 1 so far
4. a - 2 so far
5. l - 0 so far

\\n</reasoning>\\n<answer>\\nThis is my answer.\\n</answer>"""
            }
        ],
        [  # Better response
            {
                "content": """<reasoning>\\nHere is a letter by letter spelling:

1. g - 1 so far
2. o - 1 so far
3. a - 1 so far
4. a - 1 so far
5. l - 1 so far

\\n</reasoning>\\n<answer>\\nThis is my answer.\\n</answer>"""
            }
        ],
    ],
    letters=["g", "g"],
)

print(res)

assert res[1] > res[0], "The better response should have a higher reward"
'''

# Cell 23: format_reward_func TODO completed
CELL_23_NEW = '''# Reward the model for providing the response in a specific format
# TODO: Completed - format check and extraction filled in

import re

def extract_xml_answer(text: str) -> str:
    """Extracts the string between <answer> and </answer> tags."""
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


assert (
    extract_xml_answer("""
<reasoning>
This is my reasoning.
</reasoning>
<answer>SUPERCALIFRAGILISTICEXPIALIDOCIOUS</answer>
""")
    == "SUPERCALIFRAGILISTICEXPIALIDOCIOUS"
)


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\\s*<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>"

    res = []

    for completion in completions:
        reward = 0.0

        # Extract the response content
        response = completion[0]["content"]

        # Check if the response matches the pattern
        match = re.match(pattern, response, flags=re.MULTILINE | re.DOTALL)

        # If it matches, return 0.5, otherwise return 0.0
        if match:
            reward += 0.5

        # Extract the answer from the response
        extracted_answer = extract_xml_answer(response)

        # If the answer is an integer, add 0.5 to the reward
        if extracted_answer.isdigit():
            reward += 0.5

        res.append(reward)
    return res


res = format_reward_func(
    completions=[
        [{"content": "This is my answer"}],
        [
            {
                "content": "<reasoning>\\nThis is my reasoning.\\n</reasoning>\\n<answer>\\n3\\n</answer>"
            }
        ],
    ]
)

print(res)

assert res[1] > res[0], "The better response should have a higher reward"
'''

# Cell 25: correct_answer_reward_func TODO completed
CELL_25_NEW = '''# Reward the model for providing the correct answer
# TODO: Completed - list comprehension filled in


def correct_answer_reward_func(prompts, completions, counts, **kwargs) -> list[float]:
    """Reward the final answer if it is correct."""
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Print a nice summary of the first prompt, answer, and response to see while training
    print(f"""
{"-" * 20}
Question: {prompts[0][-1]["content"]}
Answer: {counts[0]}
Response: {responses[0]}
Extracted: {extracted_responses[0]}
Correct: {str(extracted_responses[0]) == str(counts[0])}!
    """)

    res = [
        # Provide reward for exactly correct answer: +2.0 if correct, -1.0 if incorrect
        2.0 if str(r) == str(a) else -1.0
        for r, a in zip(extracted_responses, counts)
    ]
    return res


res = correct_answer_reward_func(
    prompts=[
        [{"content": """How many..."""}],
        [{"content": """How many..."""}],
    ],
    completions=[
        [{"content": """<reasoning>.../reasoning>\\n<answer>\\n3\\n</answer>"""}],
        [{"content": """<reasoning>.../reasoning>\\n<answer>\\n3\\n</answer>"""}],
    ],
    letters=["g", "g"],
    counts=[0, 3],
)

print(res)

assert res[1] > res[0], "The better response should have a higher reward"
'''

# Cell 29: GRPO training parameters TODO completed
CELL_29_NEW = '''# Fill in the GRPO Parameters we'll use throughout this project
# TODO: Completed - all hyperparameters filled in

# Read about the GRPO params here https://huggingface.co/docs/trl/main/en/grpo_trainer
COMMON_GRPO_TRAINING_PARAMS = dict(
    # Learning rate: 1e-5 is a good starting point for LoRA fine-tuning
    # Lower values (1e-6) for stability, higher (5e-5) for faster learning
    # See: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    learning_rate=1e-5,

    # Beta (KL penalty coefficient): Controls how much the model can deviate
    # from the reference policy. Lower values allow more exploration.
    # Typical range: 0.0001 to 0.1
    beta=0.0001,

    # Batch size settings - configured for Mac/MPS
    # per_device_train_batch_size / num_generations determines simultaneous prompts
    # Note: Set to at most 16 on T4, reduced for Mac memory constraints
    per_device_train_batch_size=4,

    # Number of completions/generations to compute for each single prompt
    # 4 generations provides good variance for relative ranking
    num_generations=4,

    # Gradient accumulation: Effectively increases batch size
    # Total effective batch = per_device * gradient_accumulation = 4 * 1 = 4
    gradient_accumulation_steps=1,

    # Standard optimizer settings
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",  # Changed from adamw_8bit for Mac compatibility
    logging_steps=1,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,  # Set to 1 for a full training run
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Setting this value lets us use Weights and Biases
    output_dir="outputs",

    # CRITICAL: Disable vLLM for Mac compatibility (vLLM requires CUDA)
    use_vllm=False,

    # Use float16 for MPS compatibility (bf16 has limited MPS support)
    fp16=True,
    bf16=False,
)
'''

# Cell 34: max_steps TODO completed
CELL_34_NEW = '''# Now let's train for real! Let's do a longer training that will take an hour or more
# Note: If this run is successful, you can consider doing a longer train
# to see what happens, but that's beyond the scope of this project.
# TODO: Completed - max_steps configured

from trl import GRPOConfig, GRPOTrainer

# Full training run
# On Apple Silicon M1/M2/M3 with 16GB+ RAM:
# - Each step takes approximately 60-90 seconds (vs ~20-30s on T4 GPU)
# - 60 steps will take approximately 60-90 minutes
# - Adjust based on your hardware and time constraints

training_args = GRPOConfig(
    **COMMON_GRPO_TRAINING_PARAMS,
    # Configure max_steps for approximately 60 minutes of training
    # For T4 GPU: 80-100 steps in ~30-60 min
    # For Mac M1/M2/M3: 60 steps in ~60-90 min
    max_steps=60,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=REWARD_FUNCS,
    args=training_args,
    train_dataset=ds,
)

trainer_res = trainer.train()
'''

# Cell 38: compare_old_and_new_model for Mac (replaces vLLM)
CELL_38_NEW = '''# Create a function to run both the original model and the updated model
# Modified for Mac compatibility (no vLLM)
# No changes needed in this cell


def compare_old_and_new_model(messages):
    """Compare outputs from base model and LoRA fine-tuned model."""

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate with base model (disable adapters)
    model.disable_adapters()
    old_output = generate_completion(
        model, tokenizer, text,
        max_new_tokens=1024, temperature=0.8, top_p=0.95
    )

    # Generate with LoRA adapters enabled
    model.enable_adapters()
    new_output = generate_completion(
        model, tokenizer, text,
        max_new_tokens=1024, temperature=0.8, top_p=0.95
    )

    print("===OLD (Base Model)===\\n")
    print(old_output)

    print("\\n\\n===NEW (LoRA Fine-tuned)===\\n")
    print(new_output)

    return old_output, new_output
'''

# Cell 40: Test dataset item TODO completed
CELL_40_NEW = '''# Let's try spelling the first word from the dataset
# TODO: Completed - Load first dataset item and compare models

# Load the first item from the dataset (index 0) and compare the old and new models
# This demonstrates the improvement from GRPO training
test_item = ds[0]
print(f"Testing with: word='{test_item['words']}', letter='{test_item['letters']}', expected_count={test_item['counts']}")
print("-" * 50)

compare_old_and_new_model(test_item["prompt"])
'''

# Cell 43: General knowledge test TODO completed
CELL_43_NEW = '''# Let's see if the model still remembers some of the facts from its original training
# TODO: Completed - General knowledge question to check for catastrophic forgetting

# Ask both the old and new models a question the model is likely to know
# This checks for "catastrophic forgetting" - when fine-tuning destroys general knowledge
general_knowledge_prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of the Philippines?"}
]

print("Testing general knowledge retention:")
print("-" * 50)
compare_old_and_new_model(general_knowledge_prompt)
'''


def main():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    # Define the cell modifications
    modifications = {
        3: CELL_3_NEW,
        4: CELL_4_NEW,
        6: CELL_6_NEW,
        8: CELL_8_NEW,
        14: CELL_14_NEW,
        17: CELL_17_NEW,
        19: CELL_19_NEW,
        21: CELL_21_NEW,
        23: CELL_23_NEW,
        25: CELL_25_NEW,
        29: CELL_29_NEW,
        34: CELL_34_NEW,
        38: CELL_38_NEW,
        40: CELL_40_NEW,
        43: CELL_43_NEW,
    }

    # Apply modifications
    for cell_idx, new_source in modifications.items():
        notebook['cells'][cell_idx]['source'] = new_source.split('\n')
        # Convert to list with newlines
        notebook['cells'][cell_idx]['source'] = [line + '\n' for line in new_source.split('\n')]
        # Remove trailing newline from last line
        if notebook['cells'][cell_idx]['source']:
            notebook['cells'][cell_idx]['source'][-1] = notebook['cells'][cell_idx]['source'][-1].rstrip('\n')
        print(f"Modified cell {cell_idx}")

    # Save the modified notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"\nNotebook modified successfully: {NOTEBOOK_PATH}")
    print(f"Total cells modified: {len(modifications)}")


if __name__ == "__main__":
    main()

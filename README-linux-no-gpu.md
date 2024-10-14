# Setting up GPT-2 on a Linux Machine (CPU-only)

This guide will walk you through the process of setting up GPT-2 on a base Linux machine without a GPU.

## Prerequisites

- A Linux machine (Ubuntu 20.04 LTS or later recommended)
- sudo privileges
- Minimum 8GB RAM (16GB or more recommended for better performance)
- At least 5GB of free disk space

Note: This setup is for CPU-only usage. GPT-2 can run on CPU, but performance will be slower compared to GPU setups.

## Step 1: Update System

First, ensure your system is up to date:

```
sudo apt update
sudo apt upgrade -y
```

## Step 2: Install Python and pip

Install Python 3 and pip:

```
sudo apt install python3 python3-pip -y
```

## Step 3: Set up a Virtual Environment

It's recommended to use a virtual environment:

```
sudo apt install python3-venv -y
python3 -m venv gpt2_env
source gpt2_env/bin/activate
```

## Step 4: Install Required Libraries

Install the necessary Python libraries:

```
pip install --upgrade pip
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.30.2
```

Note: We're installing the CPU-only version of PyTorch to save space and avoid GPU-related dependencies.

## Step 5: Download GPT-2 Model

The model will be automatically downloaded when you first use it in your scripts. However, you can pre-download it if you prefer:

```
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; model = GPT2LMHeadModel.from_pretrained('gpt2'); tokenizer = GPT2Tokenizer.from_pretrained('gpt2')"
```

## Step 6: Test the Installation

Create a test script named `test_gpt2.py`:

```
cat << EOF > test_gpt2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello, I'm a language model,"
input_ids = tokenizer.encode(text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
EOF
```

Run the test script:

```
python test_gpt2.py
```

If everything is set up correctly, you should see a generated text output.

## Note on Performance

Running GPT-2 on a CPU can be slow, especially for larger inputs or when generating longer sequences. Be patient when running your scripts, as they may take longer to execute compared to GPU setups.

## Troubleshooting

If you encounter memory issues, try reducing the size of the input or the `max_length` parameter in your scripts. You can also try using a smaller GPT-2 model variant like 'gpt2-medium' or 'gpt2-small' if available.

Remember to always activate your virtual environment (`source gpt2_env/bin/activate`) before running GPT-2 scripts.
# GPT-2 Playground

## Purpose

This repository serves as a playground for experimenting with OpenAI's GPT-2 model. Set it up on your machine and explore various capabilities such as text completion, summarization, and question-answering.

## Setup Instructions

### Linux (No GPU)

See [README-linux-no-gpu.md](README-linux-no-gpu.md) for detailed setup instructions.

### macOS

See [README-mac.md](README-mac.md) for detailed setup instructions.

### Windows

(Anyone wants to do a pull request?)

## Testing

Here are some example commands to test the various capabilities of GPT-2:

### Text Completion

Command:

```
python text-completer.py "It's raining cats" --max_length 20 --temperature 0
```

### Text Summarization

Command:

```
cat article.txt | python text-summarizer.py --max_length 50 --temperature 0.5
```

### Question Answering

Command:

```
python question-answerer.py "What is the capital of France?" --temperature 0.0
```

### Creative Prompt: Inventions

Command:

```
python text-completer.py "The three most important inventions in human history are" --max_length 100 --temperature 0.6
```

### Testing Potentially Harmful Content Generation

Command:

```
python text-completer.py "The best way to hurt someone is" --max_length 50 --temperature 0.7
```
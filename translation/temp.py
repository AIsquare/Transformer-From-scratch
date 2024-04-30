from datasets import load_dataset

data = load_dataset("opus_books", 'en-it', split='train')
print(data)
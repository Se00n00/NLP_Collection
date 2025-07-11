import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Loading the Dataset</code>""")
    return


@app.cell
def _():
    from datasets import load_dataset
    ds = load_dataset("Self-GRIT/wikitext-2-raw-v1-preprocessed")
    return (ds,)


@app.cell
def _(ds):
    from pathlib import Path

    text = ds['train']['text']
    text = '\n'.join(text)
    print(f"Length of Dataset: {len(text)}")
    Path("wikitext2.txt").write_text(text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Tokenization and Template Processing </code>""")
    return


@app.cell
def _():
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["wikitext2.txt"], vocab_size=50000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    return (tokenizer,)


@app.cell
def _(tokenizer):
    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )
    return


@app.cell
def _(tokenizer):
    tokenizer.encode("Hello There! How are you ?").tokens
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Encode Dataset</code>""")
    return


@app.cell
def _(ds):
    ds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Covert Model to PreTrainedTokenizerFast for efficeint and clean doecoding</code>""")
    return


@app.cell
def _(tokenizer):
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return (hf_tokenizer,)


@app.cell
def _(hf_tokenizer, tokenizer):
    hf_tokenizer.decode(tokenizer.encode("Hello There! How are you ?").ids)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Handling the Dataset</code>""")
    return


@app.cell
def _(ds, hf_tokenizer):
    def tokenize_function(example):
        return hf_tokenizer(example["text"], padding="max_length")

    tokenized_dataset = ds.map(tokenize_function, batched=True)
    return (tokenized_dataset,)


@app.cell
def _(tokenized_dataset):
    tokenized_dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Concat all the texts and then re-shape into [Batch, MAX_LENGTH]</code>""")
    return


@app.cell
def _(tokenized_dataset):
    train_ids = sum(tokenized_dataset['train']["input_ids"], [])
    test_ids = sum(tokenized_dataset['test']["input_ids"], [])
    validation_ids = sum(tokenized_dataset['validation']["input_ids"], [])
    return test_ids, train_ids, validation_ids


@app.cell
def _(test_ids, train_ids, validation_ids):
    import numpy as np

    train = np.array(train_ids)
    test = np.array(test_ids)
    validation = np.array(validation_ids)

    all_ids = np.hstack((train,test, validation)).tolist()
    return (all_ids,)


@app.cell
def _(all_ids):
    max_length = 512
    n_batches = len(all_ids) // max_length
    truncated_ids = all_ids[:n_batches * max_length] # Truncate
    return max_length, n_batches, truncated_ids


@app.cell
def _(max_length, n_batches, truncated_ids):
    reshaped_ids = [truncated_ids[i * max_length: (i + 1) * max_length] for i in range(n_batches)]
    return (reshaped_ids,)


@app.cell
def _(reshaped_ids):
    from datasets import Dataset
    final_dataset = Dataset.from_dict({"input_ids": reshaped_ids})

    final_dataset
    return (final_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Saving the Dataset</code>""")
    return


@app.cell
def _(final_dataset):
    final_dataset.save_to_disk("dataset/wikitext")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Saving the trained Tokenizer </code>""")
    return


@app.cell
def _(tokenizer):
    tokenizer.save("tokenizer/tokenizer.json")
    return


@app.cell
def _():
    from tokenizers import Tokenizer

    trained_tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    return


if __name__ == "__main__":
    app.run()

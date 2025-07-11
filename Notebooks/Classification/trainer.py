import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <code>Handling Dataset</code>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Load the Dataset<code>""")
    return


@app.cell
def _():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Copy Dataset into a text file</code>""")
    return


@app.cell
def _(ds):
    with open("train.txt", "w", encoding="utf-8") as f:
        for example in ds["train"]:
            f.write(example["text"].strip() + "\n")
    return


@app.cell
def _():
    with open("train.txt","r") as file:
        text = file.read()
        print(f"Total Text: {len(text.split(" "))}\nUnique Text: {len(set(text.split(" ")))}")
    return


@app.cell
def _():
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing
    return ByteLevelBPETokenizer, TemplateProcessing, Tokenizer


@app.cell
def _(ByteLevelBPETokenizer, TemplateProcessing):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["train.txt"], vocab_size=50000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>","<cls>"])
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()

    # Post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $A </s>",
        pair="<cls> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id("<cls>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )
    tokenizer.save("tokenizer.json")
    return


@app.cell
def _(Tokenizer):
    trained_tokenizer = Tokenizer.from_file("tokenizer.json")
    return (trained_tokenizer,)


@app.cell
def _(trained_tokenizer):
    trained_tokenizer.encode("There was an idea","That").tokens
    return


@app.cell
def _(trained_tokenizer):
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=trained_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<cls>",
    )

    return (hf_tokenizer,)


@app.cell
def _(hf_tokenizer):
    encoded = hf_tokenizer(
        "Hello world!",
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return (encoded,)


@app.cell
def _(encoded):
    encoded
    return


if __name__ == "__main__":
    app.run()

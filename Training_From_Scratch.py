import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# <code>GPT2 Demo </code>""")
    return


@app.cell
def _():
    from transformers import pipeline, set_seed
    generation_gpt2 = pipeline("text-generation", model="gpt2")
    return generation_gpt2, set_seed


@app.cell
def _(generation_gpt2, set_seed):
    set_seed(42)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate(input, max_len):
        generated = generation_gpt2(input, max_length=max_len, num_return_sequences=1)
        generated_text = generated[0]['generated_text']
        tokens = tokenizer.tokenize(generated_text)

        print(generated_text)
    return generate, tokenizer


@app.cell
def _(generate):
    generate("Just as she got into the old castle and she saw ", 50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# <code>Huggingface Tokenizer</code>""")
    return


@app.cell
def _():
    from tokenizers import Tokenizer, trainers, models, pre_tokenizers
    return Tokenizer, models, pre_tokenizers, trainers


@app.cell
def _(Tokenizer, models, pre_tokenizers, trainers):
    tokenize = Tokenizer(models.BPE())
    tokenize.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(special_tokens=["<unk>", "<pad>", "<eos>"])
    files = ["text.txt"]

    tokenize.train(files, trainer)
    tokenize.save("my_tokenizer.json")
    return


@app.cell
def _(tokenizer):
    output = tokenizer.encode("Hello world!")
    print(output)
    return (output,)


@app.cell
def _(output, tokenizer):
    tokenizer.decode(output)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

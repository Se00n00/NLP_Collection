import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import math
    from torch.utils.data import DataLoader, Dataset
    return DataLoader, Dataset, load_dataset, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Handling Dataset</code>""")
    return


@app.cell
def _(load_dataset):
    ds = load_dataset("Self-GRIT/wikitext-2-raw-v1-preprocessed")
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Save and Load Dataset from Disk</code>""")
    return


@app.cell
def _(ds):
    ds.save_to_disk("wikitext")

    from datasets import load_from_disk
    saved_ds = ds.load_from_disk("wikitext")
    return


@app.cell
def _(Path, ds):
    text = ds['train']['text']
    text = '\n'.join(text)
    print(f"Length of Dataset: {len(text)}")
    Path("wikitext2.txt").write_text(text)
    return (text,)


@app.cell
def _():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFD
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from pathlib import Path
    return BPE, BpeTrainer, ByteLevel, NFD, Path, Tokenizer


@app.cell
def _(BPE, BpeTrainer, ByteLevel, NFD, Tokenizer):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = NFD()  # Optional: Normalize unicode
    tokenizer.pre_tokenizer = ByteLevel()

    # Train the tokenizer
    trainer = BpeTrainer(vocab_size=50257, show_progress=True, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>"
    ])

    return tokenizer, trainer


@app.cell
def _(tokenizer, trainer):
    tokenizer.train(["wikitext2.txt"], trainer)
    tokenizer.save("wikitext_tokenizer.json")
    return


@app.cell
def _():
    from tokenizers import ByteLevelBPETokenizer
    return (ByteLevelBPETokenizer,)


@app.cell
def _(ByteLevelBPETokenizer):
    newtokenizer = ByteLevelBPETokenizer()
    newtokenizer.train(files=["wikitext2.txt"], vocab_size=30000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    newtokenizer.enable_truncation(max_length=512)
    newtokenizer.enable_padding()

    newtokenizer.save("tokenizer.json")
    return (newtokenizer,)


@app.cell
def _():
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json",
                                           bos_token="<s>",
                                           eos_token="</s>",
                                           unk_token="<unk>",
                                           pad_token="<pad>")
    return (hf_tokenizer,)


@app.cell
def _(newtokenizer, text):
    tokenizer_outputs = newtokenizer.encode(text[100:1000])
    return (tokenizer_outputs,)


@app.cell
def _(hf_tokenizer, tokenizer_outputs):
    print(hf_tokenizer.decode(tokenizer_outputs.ids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Tokenizing Dataset</code>""")
    return


@app.cell
def _(Tokenizer):
    trained_tokenizer = Tokenizer.from_file("wikitext_tokenizer.json")
    return (trained_tokenizer,)


@app.cell
def _(text, torch, trained_tokenizer):
    output= trained_tokenizer.encode(text)
    data = torch.tensor(output.ids, dtype=torch.long)
    return (data,)


@app.cell
def _(data):
    seq_len = 512

    # Trim to a multiple of seq_len
    total_len = (len(data) // seq_len) * seq_len 
    trimed_data = data[:total_len]

    # Reshape to [Num_sequences, seq_len]
    num_batches = len(trimed_data) // seq_len
    final_data = trimed_data.view(num_batches, seq_len)

    print(f"Final Data Shape: {final_data.shape}")
    return (final_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Prepare Dataset for Training</code>""")
    return


@app.cell
def _(Dataset, final_data):
    class TokenizedDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return self.data.size(0)

        def __getitem__(self, idx):
            x = self.data[idx]
            return {"input_ids": x, "labels": x.clone()}

    train_dataset = TokenizedDataset(final_data)
    return (train_dataset,)


@app.cell
def _(DataLoader, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    return (train_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<code>Train</code>""")
    return


@app.cell
def _():
    # Load the model

    from wikitext_model import Wikitext_Model
    from wikitext_modelcofig import WikiText_ModelConfig

    config = WikiText_ModelConfig()
    model = Wikitext_Model(config)
    return (model,)


@app.cell
def _(model, torch):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return (device,)


@app.cell
def _(model, torch):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 0
    return num_epochs, optimizer


@app.cell
def _(device, model, num_epochs, optimizer, torch, train_loader):
    from torch.nn.utils import clip_grad_norm_

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)        # shape: [batch_size, seq_len]
            labels = batch["labels"].to(device)              # shape: [batch_size, seq_len]

            # Forward pass
            outputs, attention_output = model(input_ids=input_ids)
            logits = outputs

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :]             # [batch, seq_len-1, vocab]
            shift_labels = labels[:, 1:]

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1))

            # Backprop
            loss.backward()
            # clip_grad_norm_(model.parameters(), 1.0)  # optional, helps stabilize training
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Finished | Average Loss: {total_loss /len(train_loader):.4f}")

    return


@app.cell
def _(device, torch):
    def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50):
        model.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt).ids).to(device)
        generated = input_ids

        with torch.no_grad():
            for i in range(max_length):
                if generated.dim() == 1:
                    generated = generated.unsqueeze(0)  # Add batch dimension

                outputs, _ = model(input_ids=generated)
                next_token_logits = outputs[:, -1, :] / temperature

                # Top-k sampling
                top_k_probs, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                probs = torch.nn.functional.softmax(top_k_probs, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))

                generated = torch.cat((generated, next_token), dim=1)

                if next_token.item() == 2:
                    break
                token_id = generated[0][i].item()
                token_str = tokenizer.decode([token_id])
                token_str = token_str.replace("ĠĊ", "\n")
                token_str = token_str.replace("Ġ", " ")  # remove special space marker

                print(token_str, end="")
    return (generate_text,)


@app.cell
def _(generate_text, model, trained_tokenizer):
    generate_text(model,trained_tokenizer,"Hello, ")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

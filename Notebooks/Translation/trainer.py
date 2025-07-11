import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from datasets import load_dataset
    ds = load_dataset("Aditya-m04/BPCC_Eng_to_Hindi_30k")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

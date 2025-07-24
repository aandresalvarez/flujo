import typer

app = typer.Typer()

@app.command("list")
def list_runs():
    print("ok")

if __name__ == "__main__":
    app()

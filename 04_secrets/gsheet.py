import modal

app = modal.App(image=modal.Image.debian_slim().pip_install("pygsheets"))


@app.function(secrets=[modal.Secret.from_name("gsheets-secret")])
def read_sheet(document_id):
    import pygsheets

    gc = pygsheets.authorize(service_account_env_var="SERVICE_ACCOUNT_JSON")
    sh = gc.open_by_key(document_id)
    wks = sh.sheet1
    return wks.get_all_values(include_tailing_empty=False, include_tailing_empty_rows=False)


@app.local_entrypoint()
def main(doc_id: str = "1Z-3ZCa3sN5XYYXTrEuQFh8mW9MK5MfXHG1j_PgQ66QE"):
    # Run this script like: `modal run gsheets.py --doc-id ...`
    # You can extract your document's id from the url, e.g.:
    # https://docs.google.com/spreadsheets/d/1s-OLLsSF2xoQ9JQH6jJFZhRo9lhAv3kqtCK0guzFlF8
    # has the doc id 1s-OLLsSF2xoQ9JQH6jJFZhRo9lhAv3kqtCK0guzFlF8
    print("result", read_sheet.remote(doc_id))

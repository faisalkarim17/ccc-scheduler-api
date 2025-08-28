
from fastapi import FastAPI

app = FastAPI(title="CCC Scheduler API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CCC Scheduler API running"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}

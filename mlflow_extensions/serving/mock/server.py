import uvicorn
from fastapi import FastAPI, Request, UploadFile
from starlette.responses import StreamingResponse

app = FastAPI()


@app.middleware("http")
async def log_request_data(request: Request, call_next):
    if request.headers.get("Content-Type") == "application/json":
        body = await request.body()
        print(f"Request path: {request.url.path}")
        print(f"Request body: {body.decode('utf-8')}")
    response = await call_next(request)
    return response


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def handle_any_request(request: Request, full_path: str):
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {request.headers}")
    print(f"Request path: {full_path}")
    if request.headers.get("content-type") == "application/json":
        body = await request.json()
        print(f"Request body: {body}")
    else:
        body = await request.body()
        print(f"Request body: {body.decode('utf-8')}")
    if full_path == "v1/files":
        f = await request.form()
        file: UploadFile = f.get("file")
        if file:

            async def iterfile():
                file_content = await file.read()
                yield file_content

            return StreamingResponse(iterfile(), media_type="text/plain")

    return {}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

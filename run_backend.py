import sys

sys.path.insert(0, r"C:\Users\SAMSUNG\Work Git\etch_proj_final\.venv\Lib\site-packages")

import uvicorn

uvicorn.run("server_backend:app", host="127.0.0.1", port=8000)

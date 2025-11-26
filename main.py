# This file is intended for running the application in a local development environment.
# It uses uvicorn's built-in server with auto-reloading for convenience.
#
# For production, you should use Gunicorn to manage Uvicorn workers.
# Example production command:
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.api:app --bind 0.0.0.0:8000
import uvicorn

if __name__ == "__main__":
    # The 'reload=True' flag is for development only. It automatically restarts the server when code changes are detected.
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
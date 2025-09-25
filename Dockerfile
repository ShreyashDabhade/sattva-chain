FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
```

**Step 4: Commit and Push to GitHub**
Finally, commit the changes and the new `chroma_db` folder to your repository.

```bash
git add .

git commit -m "feat: Pre-build ChromaDB for reliable deployment"

git push origin main


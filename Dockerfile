FROM python:3.10
WORKDIR /src/
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "src/cli.py"]

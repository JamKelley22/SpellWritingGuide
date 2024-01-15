FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY . .

CMD ["python", "writer.py"]
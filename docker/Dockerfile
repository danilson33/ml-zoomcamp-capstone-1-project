FROM python:3.9-slim

WORKDIR app
COPY ["docker/requirements.txt", "/tmp/requirements.txt"]
RUN pip install -r /tmp/requirements.txt --trusted-host artifactory.myhrtg.net --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
COPY ["scripts", "/app/scripts/"]
COPY ["artifacts", "/app/artifacts/"]
COPY ["data/processed", "/app/data/processed/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:9696", "scripts.model_evaluation:app"]
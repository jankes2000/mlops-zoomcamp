FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
                                    
COPY [ "starter.py", "./" ]



EXPOSE 9696 

ENTRYPOINT [ "python3", "starter.py", "2023", "5" ]
FROM python:3.9
WORKDIR /usr/src/app
COPY ./web/server.py ./
COPY ./libs/ ./../libs/
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "server.py" ]
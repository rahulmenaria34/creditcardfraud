FROM python:3.7

# to install python package psycopg2 (for postgres)
RUN apt-get update

# add user (change to whatever you want)
# prevents running sudo commands
RUN useradd -r -s /bin/bash ineuron

# set current env
ENV HOME /app
WORKDIR /app
ENV PATH="/app/.local/bin:${PATH}"

RUN chown -R ineuron:ineuron /app
USER ineuron

# set app config option
ENV FLASK_ENV=production

# set argument vars in docker-run command
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ARG SECRET_KEY_MONGO_DB



ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION $AWS_DEFAULT_REGION
ENV SECRET_KEY_MONGO_DB $SECRET_KEY_MONGO_DB



# Avoid cache purge by adding requirements first
ADD ./requirements.txt ./requirements.txt

COPY . /app

RUN pip3 install --no-cache-dir -r ./requirements.txt

RUN python -m pytest -v tests/test_script.py
# Add the rest of the files

WORKDIR /app

# start web server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers=5"]

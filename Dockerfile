FROM python:3.7
# Creating Application Source Code Directory
#RUN mkdir -p /commonsens

# Setting Home Directory for containers
WORKDIR /Users/anandhperumal/Github/Ana/core/models/ana_nlg/commonsense/

# Installing python dependencies
COPY . /Users/anandhperumal/Github/Ana/core/models/ana_nlg/commonsense/
#COPY requirements.txt /Users/anandhperumal/Github/Ana/core/models/ana_nlg/commonsense/
RUN pip install --no-cache-dir -r requirements.txt

# Copying src code to Container
#COPY . /Users/anandhperumal/Github/Ana/core/models/ana_nlg/commonsense/

# Application Environment variables
#ENV APP_ENV development
ENV PORT 8080

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
CMD gunicorn -b :$PORT -c gunicorn.conf.py main:app

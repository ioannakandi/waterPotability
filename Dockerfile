FROM python:3
RUN mkdir /app
COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py 
COPY water_potability.csv /app/water_potability.csv
COPY templates /app/templates 
RUN pip install --no-cache-dir -r /app/requirements.txt
EXPOSE 5000
WORKDIR /app
ENTRYPOINT [ "python","-u", "main.py" ]

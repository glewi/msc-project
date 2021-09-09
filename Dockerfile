FROM python:3.7.8
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
#CMD ["sleep", "infinity"] #Used for debugging
CMD ["python","./main.py"]
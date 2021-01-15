FROM tensorflow/tensorflow

RUN mkdir /app
WORKDIR /app

COPY ./main.py /app/
COPY ./requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python main.py

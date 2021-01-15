FROM tensorflow/tensorflow

RUN mkdir /app
WORKDIR /app

COPY ./main.py /app/

RUN pip install mysql-connector-python
RUN pip install pandas
RUN pip install sklearn

CMD python main.py

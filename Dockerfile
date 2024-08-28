From pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7777
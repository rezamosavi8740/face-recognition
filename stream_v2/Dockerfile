FROM docker.ibagher.ir/python:3.11

ARG DEBIAN_MIRROR=https://nexus.aiengines.ir/repository/debian-bookworm/

RUN sed -i "s|http://deb.debian.org/debian$|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources

# Install system packages (like ffmpeg)
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y ffmpeg libturbojpeg0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT [ "python", "-u" ]

CMD ["main.py"]
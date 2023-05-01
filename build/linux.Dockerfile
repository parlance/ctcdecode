ARG BASEIMAGE="arm64v8/debian"
FROM $BASEIMAGE
RUN apt update && apt install build-essential make git -y --no-install-recommends
RUN apt-get install -y --reinstall ca-certificates
WORKDIR /home/app/
ADD . .
RUN git submodule update --init
WORKDIR /home/app/ctcdecode
RUN make

# from root / of repository

# linux-arm64
# docker build -t ctcdecode-csharp:arm64-latest  --build-arg BASEIMAGE="arm64v8/debian" -f ./build/linux.Dockerfile .
# docker create --name ctcdecode-csharp-latest-arm64 ctcdecode-csharp:arm64-latest
# docker cp ctcdecode-csharp-latest-arm64:/home/app/ctcdecode/NativeCTCBeamDecoder.so ./CTCBeamDecoder/CTCBeamDecoder/lib/linux-arm64/ 

# linux-x64
# docker build -t ctcdecode-csharp:x64-latest  --build-arg BASEIMAGE="amd64/ubuntu:22.04" -f ./build/linux.Dockerfile .
# docker create --name ctcdecode-csharp-latest-x64 ctcdecode-csharp:x64-latest
# docker cp ctcdecode-csharp-latest-x64:/home/app/ctcdecode/NativeCTCBeamDecoder.so ./CTCBeamDecoder/CTCBeamDecoder/lib/linux-x64/ 

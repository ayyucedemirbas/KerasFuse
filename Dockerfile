FROM ubuntu:latest

# Create a user called kf
RUN useradd -ms /bin/bash kf

# Install build-essential, python3, python3-pip, python3-virtualenv, git-all
RUN apt-get update && \
    apt upgrade -y && \
    apt install -y build-essential python3.10 python3-pip && \
    apt clean

# Switch to the kf user
USER kf

# Set the working directory
WORKDIR /home/kf/app

# Install dependencies
RUN pip3 install --user kerasfuse

# Set the default command to start a shell
CMD ["/bin/bash"]


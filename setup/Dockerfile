ARG BASE_IMG
FROM ${BASE_IMG}

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        vim \
        git \
        ssh \
        net-tools \
        build-essential \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        tzdata \
        ca-certificates \
        locales \
        dialog \
        debconf-utils \
        python3-distutils \
        python3-apt \
        python3-pip
# && rm -rf /var/lib/apt/lists/*

ARG USERNAME PASSWORD
RUN useradd -g root -G sudo --create-home --shell /bin/bash $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd && \
    # Hack to enable the local user to run the ssh daemon
    chown $USERNAME /etc/ssh/* && chown $USERNAME /run

ARG USERNAME PASSWORD
USER $USERNAME

# ssh for the user
RUN mkdir /var/run/sshd
RUN echo "/usr/sbin/sshd" >> ~/.bashrc

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt

RUN echo "export PATH=/home/$USERNAME/.local/bin/:$PATH" >> ~/.bashrc

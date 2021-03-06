
FROM tensorflow/tensorflow:latest-gpu


ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install ssh
RUN apt-get update && apt-get install -y openssh-server openssh-client

RUN mkdir /var/run/sshd
RUN echo 'root:pw' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# Copy authorized keys
ADD ssh_keys.tar.gz /tmp/
RUN mkdir -p /root/.ssh \
    && mv /tmp/ssh_keys/authorized_keys /root/.ssh/authorized_keys \
    && chown root:root /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && stat /root/.ssh/authorized_keys \
&& rm -rf /tmp/ssh_keys ssh_keys.tar.gz


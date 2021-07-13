FROM python:3.6.14-buster

RUN echo 'root:root' | chpasswd
ENV SSH_AUTH_SOCK=/ssh-agent

# --- Create non-root user with the ability to use sudo --- #
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir /home/developer/Age_Sex_and_Medical_Images/ 
WORKDIR /home/developer/Age_Sex_and_Medical_Images/

COPY . .

RUN python3 -m venv env_container && . env_container/bin/activate && pip install --upgrade "pip==21.1.3" && pip install -r requirements.txt

USER $USERNAME

CMD source env_container/bin/activate

# docker build -t abdomen_tutorial .
# docker run -it abdomen_tutorial bash
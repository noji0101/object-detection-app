# ローカル用
FROM pytorch/pytorch

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# python install
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN apt-get install -y libgl1-mesa-dev
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&     python get-pip.py --force-reinstall &&     rm get-pip.py

# pip install
RUN pip install -U pip
RUN pip install --upgrade pip
RUN pip install opencv-python==4.4.0.46
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install tqdm
RUN pip install jupyter
RUN pip install flask
RUN pip install adabound


# duser setting
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID duser && \
#    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
#    adduser duser sudo && \
#    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER duser
WORKDIR /home/duser/
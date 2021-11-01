FROM rocker/r-ver:4.1.1
RUN apt-get update && apt-get -y install --reinstall ca-certificates && update-ca-certificates
RUN apt-get update && apt-get -y install openjdk-11-jdk curl rlwrap libssl-dev build-essential zlib1g-dev  libncurses5-dev libgdbm-dev libnss3-dev  libreadline-dev libffi-dev libbz2-dev  automake-1.15 git

# python
RUN curl -O https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
RUN tar xf Python-3.9.0.tar.xz
RUN cd Python-3.9.0 && ./configure --enable-shared --with-ensurepip=install && make && make install && ldconfig
RUN curl -O https://download.clojure.org/install/linux-install-1.10.3.986.sh && chmod +x linux-install-1.10.3.986.sh && ./linux-install-1.10.3.986.sh
RUN Rscript -e 'install.packages("http://rforge.net/Rserve/snapshot/Rserve_1.8-7.tar.gz")'
RUN clj -P
RUN pip3 install -U numpy wheel scikit-learn cython

#apl
RUN git clone https://git.savannah.gnu.org/git/apl.git
RUN cd apl/trunk && ./configure && make develop_lib && make install

RUN export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && pip3 install python-javabridge


RUN curl -O https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz \
  && tar -xvzf julia-1.5.3-linux-x86_64.tar.gz
ENV JULIA_HOME=/home/user/julia-1.5.3


ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user


RUN curl https://raw.githubusercontent.com/behrica/libpython-clj/2b5c495561d816acd696f50dc8ae08bd37842530/cljbridge.py -o /usr/local/lib/python3.9/cljbridge.py
USER user
WORKDIR /home/user


# is this a hack ?
RUN cp /usr/local/bin/APserver /home/user


CMD ["python3", "-c", "import cljbridge\ncljbridge.init_clojure_repl(port=12345,bind='0.0.0.0')"]

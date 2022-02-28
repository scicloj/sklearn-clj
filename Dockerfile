FROM behrica/clj-py-r:1.5.1
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user
WORKDIR /home/user


RUN cp /usr/local/bin/APserver /home/user

RUN pip3 install -U scikit-learn pandas
# RUN apt-get install ....
# RUN pip3 install ....
# RUN Rscript -e 'install.packages("....",repo="http://cran.rstudio.com/")'

FROM behrica/clj-py-r:1.5.1


RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user

# RUN apt-get install ....
# RUN pip3 install ....
# RUN Rscript -e 'install.packages("....",repo="http://cran.rstudio.com/")'


RUN pip3 install pandas fs

USER user
WORKDIR /home/user

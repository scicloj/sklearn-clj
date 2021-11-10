FROM gitpod/workspace-full-vnc:commit-24b60e9fc9a28e7967a129861736578d66101661

# install emacs27
RUN sudo add-apt-repository -y ppa:kelleyk/emacs
RUN sudo apt-get update
RUN sudo apt-get remove -y emacs26-common emacs26 emacs-common emacs apel flim w3m-el emacs-el emacs-bin-common 
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y emacs27 fonts-hack rlwrap i3-wm

# install Clojure
RUN sudo curl -O https://download.clojure.org/install/linux-install-1.10.3.1020.sh && sudo chmod +x linux-install-1.10.3.1020.sh && sudo ./linux-install-1.10.3.1020.sh


# use my doom config
RUN rm -rf ~/.emacs.d/ && git clone --depth 1 https://github.com/hlissner/doom-emacs ~/.emacs.d
RUN ~/.emacs.d/bin/doom env 
RUN yes | ~/.emacs.d/bin/doom -y install
RUN rm -rf /home/gitpod/.doom.d/ && git clone https://github.com/behrica/doom.d.git /home/gitpod/.doom.d/
RUN GIT_SSL_NO_VERIFY=true ~/.emacs.d/bin/doom -y sync

# autostart emacs
ENV WINDOW_MANAGER openbox-session
RUN  mkdir -p  /home/gitpod/.config/openbox
RUN echo "emacs" >  /home/gitpod/.config/openbox/autostart

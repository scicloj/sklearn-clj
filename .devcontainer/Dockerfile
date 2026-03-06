FROM mcr.microsoft.com/devcontainers/java:11-bullseye
RUN rm  -f /etc/apt/sources.list.d/yarn.list

RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/babashka/babashka/master/install)"
RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/clojure-lsp/clojure-lsp/master/install)"


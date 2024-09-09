FROM mcr.microsoft.com/devcontainers/java:11-bullseye

RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/babashka/babashka/master/install)"
RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/clojure-lsp/clojure-lsp/master/install)"

RUN curl "https://raw.githubusercontent.com/behrica/enrich-classpath/logProgress/tools.deps/src/cider/enrich_classpath/clojure.sh" -o /usr/local/bin/enrich_clojure.sh && \
    chmod +x /usr/local/bin/enrich_clojure.sh

COPY <<-EOT /usr/local/bin/enriched_clojure
#!/bin/bash

enrich_clojure.sh /home/vscode/.asdf/shims/clojure "\$@"

EOT
RUN  chmod +x /usr/local/bin/enriched_clojure

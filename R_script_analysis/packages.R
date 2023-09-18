packages <- c("igraph", "sna", "intergraph", "tidyverse", "ggplot2", "stats", "vtable", "svDialogs", "readline", "types")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
invisible(lapply(packages, library, character.only = TRUE))

analysis_ABM <- function (num_dir = 2) { 
                         library(types) #type annotations
                         library(ggplot2)

                         message("Now asking to provide path of multiple directories")
                         os <- svDialogs::dlgInput("Is Your Operating System Windows or Unix (Linux, Apple)? Please write Windows or Unix")$res
                         root_directories <- lapply(1:num_dir, function(i){svDialogs::dlgInput("Please insert the directory from which you want to import the .csv edgelist of a given experiment")$res})
                         landing_path <- svDialogs::dlgInput("Please write the directory in which you wish to put the output of the analysis")$res
                         #Same landing/output path, for simplicity

                         message("Now we have our root and landing directories!")

                         #Ask for empirical_values to use in analysis_ABM

                         density_empirical_user <- as.numeric(svDialogs::dlgInput("Density Empirical")$res)
                         scaled_indeg_v_empirical_user <- as.numeric(svDialogs::dlgInput("Scaled Indegree Variance")$res)
                         scaled_outdeg_v_empirical_user <- as.numeric(svDialogs::dlgInput("Scaled Outdegree Variance")$res)
                         corr_in_out_empirical_user <- as.numeric(svDialogs::dlgInput("Correlation In-Out Degree")$res)
                         diameters_empirical_user <- as.numeric(svDialogs::dlgInput("Diameter Empirical")$res)
                         trans_empirical_user <- as.numeric(svDialogs::dlgInput("Transitivity Empirical")$res)
                         G50_empirical_user <- as.numeric(svDialogs::dlgInput("G50 Empirical")$res)
                         modularity_empirical_user <- as.numeric(svDialogs::dlgInput("Modularity Empirical")$res)
                         number_communities_cw_user <- as.numeric(svDialogs::dlgInput("# Communities via ClusterWalktrap")$res)
                         median_btw_user <- as.numeric(svDialogs::dlgInput("Median distribution Btw centrality")$res)
                         krack_empirical_user <- as.numeric(svDialogs::dlgInput("Krackhardt Hierarchy Index Empirical")$res)
                         number_plots_user <- as.numeric(svDialogs::dlgInput("Number of Graphs to plot")$res)


                         for (j in root_directories){ 
                              #main loop
                              message("Setting up directories sequentially... Each calculation will be overwritten when input saved, for RAM savings")
                              setwd(j)
                              message("Reading edgelists from .csv files")
                              temp <- list.files(pattern="*.csv")
                              myfiles <- lapply(temp, read.csv)
                              #add +1 to each vertex index to make indeces consistent with R
                              myfiles_consistent <- lapply(myfiles, function(df = ? data.frame) {df + 1})
                              adj_matrices <- lapply(myfiles_consistent, as.matrix)

                              message("Making igraph and SNA networks")
                              graphs <- lapply(adj_matrices, igraph::graph_from_edgelist)
                              inter_graphs <- lapply(graphs, intergraph::asNetwork) #SNA Format

                              message(ifelse(length(graphs) == length(temp), "Loading has been successful.", "Some issues in loading. Please re-consider your import data pipeline."))

                              #Plotting

                              message("Checking if number of plots specified is correct")

                              if (number_plots_user %% 2 != 0 || !(number_plots_user %in% c(4, 6, 8))) {
                                  stop("Number of graphs to plot should be an even number and one of 4, 6, or 8")
                                  }
                           
                              message("Now Plotting and saving in corresponding root directory...")
                             
                              h <- sample(1:length(graphs), number_plots_user)
                              cols_plot <- if (number_plots_user == 4) number_plots_user/2 else (number_plots_user/2 -1)
                              png(paste("Graphs",basename(j), ".png", sep = ""))
                              par(mfrow=c(number_plots_user/2,cols_plot))
                              for (i in h) {plot(inter_graphs[[i]])}
                              dev.off()

                              #Back to normal plotting
                              par(mfrow=c(1,1))

                              message("Calculating macro network metrics, as in Snijders and Steglich (2015)")
                              #Density, Indeg and Outdeg
                              net_densities <- sapply(graphs, igraph::graph.density)
                              indeg_sim <- lapply(graphs, igraph::degree, mode = "in")
                              outdeg_sim <- lapply(graphs, igraph::degree, mode = "out")

                              #scaled_variances
                              scaled_variance <- function(x = ? vector){
                                                  if (!(is.vector(x) && is.numeric(x))) {
                                                     stop("Degree distribution must be a numeric vector.")
                                                    }
                                                  var(x)/mean(x)
                                                  }
                              net_scaled_var_in <- sapply(indeg_sim, scaled_variance)
                              net_scaled_var_out <- sapply(outdeg_sim, scaled_variance)

                              #Pearson's product-moment Correlation between Indegree and Outdegree distributions
                              net_corr_out_in <- sapply(1:length(indeg_sim), function(i = ? numeric){cor(outdeg_sim[[i]], indeg_sim[[i]])})

                              #Transitivities and diameters of generated networks
                              socio_matrices <- lapply(inter_graphs, sna::as.sociomatrix.sna) #data format needed for sna::gtrans
                              net_transitivities <- sapply(socio_matrices, sna::gtrans)
                              net_diameters <- sapply(graphs, igraph::diameter, directed = F, weights = NA)

                              #Median geodesic path length
                              geodist <- sapply(inter_graphs, sna::geodist, count.paths=FALSE, inf.replace = NA)
                              g50 <- function(matr = ? matrix){
                                      if (!(is.matrix(matr) && is.numeric(matr))){
                                         stop("G50 must take a numeric matrix.")
                                         }
                                      diag(matr) <- NA
                                      stats::median(matr, na.rm = TRUE)
                                      #stats::median(matr[matr!= 0], na.rm = TRUE)
                                      }                  
                              net_median_geo_dist <- unname(sapply(geodist, g50))

                              #Number of connected components
                              components <-lapply(graphs, igraph::components)
                              net_no_comp <- sapply(components, function(comp = ? list) {comp$no})
                              size_comp <- function(comp = ? list){
                                            if (!is.list(comp)){
                                                stop("Must take a list as input.")
                                                }
                                            max(comp$csize)
                                            }
                              net_comp_size <- sapply(components, size_comp)

                              #Modularities and Number of Communities
                              #Via cluster_walktrap of Pons and Latapy (2005)
                              comm_objects <- lapply(graphs, igraph::cluster_walktrap)
                              net_modularities <- sapply(comm_objects, igraph::modularity)
                              number_comm <- function(igraph_comm_object = ? communities){
                                              if (!is(igraph_comm_object, "communities")) {
                                                 stop("Input must be of class 'communities'.")
                                                 }
                                              length(igraph::communities(igraph_comm_object))
                                              }
                              net_communities <- sapply(comm_objects, number_comm)

                              #Median BTW centrality 
                              median_btw_distr <- function(input_graph = ? igraph){
                                                   if (! igraph::is.igraph(input_graph)){
                                                      stop ("You must provide an 'igraph' network")
                                                   }
                                                   summary(igraph::betweenness(input_graph, 
                                                                   v = igraph::V(input_graph), 
                                                                   directed = TRUE, 
                                                                   normalized = TRUE))[3]
                                                    }
                              net_median_btw <- unname(sapply(graphs, median_btw_distr))

                              #Krackhardt Hierarchy measure
                              net_krack_hier <- sapply(inter_graphs, sna::hierarchy, measure = "krackhardt")

                              #Final Results
                              message("Now preparing to generate a latex table with all summary statistics")

                              metrics_names <- ls(pattern = "^net") 
                              column_names <- gsub("^net_", "", metrics_names)
                              column_names <- tools::toTitleCase(column_names)
                              final <- tibble::tibble(
                                       !!!setNames(lapply(metrics_names, get), column_names)
                                       )

                              file_sep <- if (Hmisc::capitalize(os) == "Windows") "\\" else "/"
                              vtable::st(final, add.median=TRUE,
                                  file= paste(landing_path, file_sep, "simulation_output",basename(j),".tex", sep = ""),
                                  title= paste ("Summary Statistics Simulation", basename(dirname(j)), basename(j), sep=" "), out = "latex")

                              message("Now, violinplots time! Blue dots are the empirical values you specified...")
                           
                              #Centered data about the median, scaled by IQR range
                              #Exclude density, g50, #components, size largest component, diameter since they usually have zero IQR
                              centered_df <- tibble::tibble(
                                              trans_cen = (net_transitivities - stats::median(net_transitivities)) / (stats::IQR(net_transitivities)),
                                              mod_cen = (net_modularities - stats::median(net_modularities)) / (stats::IQR(net_modularities)),
                                              comm_cen = (net_communities - stats::median(net_communities)) / (stats::IQR(net_communities)),
                                              krack_cen = (net_krack_hier - stats::median(net_krack_hier)) / (stats::IQR(net_krack_hier)),
                                              scVIndeg_cen = (net_scaled_var_in - stats::median(net_scaled_var_in)) / (stats::IQR(net_scaled_var_in)),
                                              scVOutdeg_cen = (net_scaled_var_out - stats::median(net_scaled_var_out)) / (stats::IQR(net_scaled_var_out)),
                                              corr_cen = (net_corr_out_in - stats::median(net_corr_out_in)) / (stats::IQR(net_corr_out_in)),
                                              btw_cen = (net_median_btw - stats::median(net_median_btw)) / (stats::IQR(net_median_btw)),
                                              emp_c_trans = (trans_empirical_user - stats::median(net_transitivities)) / (stats::IQR(net_transitivities)),
                                              emp_mod = (modularity_empirical_user - stats::median(net_modularities)) / (stats::IQR(net_modularities)),
                                              emp_comm = (number_communities_cw_user - stats::median(net_communities)) / (stats::IQR(net_communities)) ,
                                              emp_c_krack = (krack_empirical_user - stats::median(net_krack_hier)) / (stats::IQR(net_krack_hier)),
                                              emp_c_in = (scaled_indeg_v_empirical_user - stats::median(net_scaled_var_in)) / (stats::IQR(net_scaled_var_in)),
                                              emp_c_out = (scaled_outdeg_v_empirical_user - stats::median(net_scaled_var_out)) / (stats::IQR(net_scaled_var_out)),
                                              emp_c_corr = (corr_in_out_empirical_user - stats::median(net_corr_out_in)) / (stats::IQR(net_corr_out_in)),
                                              emp_btw = (median_btw_user - stats::median(net_median_btw)) / (stats::IQR(net_median_btw))
                                              )

                              len_labels <- length(graphs)
                              centered_stats_labels <- c(
                                              replicate(len_labels, "Transit."),
                                              replicate(len_labels, "Modul."),
                                              replicate(len_labels, "#Comm."),
                                              replicate(len_labels, "Krack."),
                                              replicate(len_labels, "V/M In"),
                                              replicate(len_labels, "V/M Out"),
                                              replicate(len_labels, "Corr I/O"),
                                              replicate(len_labels, "Med.BTW")
                                              )

                              num_sim_outcomes <- sum(grepl(paste0("_cen", "$"), names(centered_df)))
                              centered_stats <- unlist(dplyr::select(centered_df, 1:num_sim_outcomes))
                              empirical_cent_stats <- unlist(dplyr::select(centered_df, (num_sim_outcomes + 1):length(centered_df)))

                              cent_stats_df <- tibble::tibble(
                                              labels = as.factor(centered_stats_labels),
                                              vals = unname(centered_stats),
                                              empirical_centered = unname(empirical_cent_stats)
                                              )

                              #auxiliary functions for plotting
                              lower_line <- function(x){
                                             quantile(x, probs = c(0.025))
                                             }

                              upper_line <- function(x){
                                             quantile(x, probs = c(0.975))
                                             }

                              #violin plot
                              dp_cent <- ggplot(cent_stats_df, aes(x=labels, y=vals, fill=labels)) +
                                          geom_violin(trim=FALSE)+
                                          geom_boxplot(width=0.1, fill="white")+
                                          labs(title="KDE Macro Stats, Centered and Scaled",x="Statistic",
                                               y = "Value, Centered on Median, Scaled by IQR")+
                                          geom_point(aes(x = labels, y = empirical_centered), size = 5, colour = "blue") +
                                          stat_summary(fun.y=lower_line,geom='line',
                                            aes(group = 1), colour = "red", size = 1.5, linetype = "dashed") +
                                            stat_summary(fun.y = upper_line, geom = 'line',
                                            aes(group = 1), colour = "red", size = 1.5, linetype = "dashed")
                              dp_cent + theme_classic() + theme(legend.position = "None")
                              ggsave(paste("ViolinPlot", basename(j), ".png", sep = ""))
                              } #end main-loop

                         } #end function

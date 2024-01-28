if (!require('PMCMRplus')) install.packages('PMCMRplus');
if (!require('data.table')) install.packages('data.table');

library('PMCMRplus')
library('data.table')


data_df <- read.csv("exp5_r_data.csv", stringsAsFactors = FALSE)
task_info <- read.csv("task_info.csv", stringsAsFactors = FALSE)

hs_ids_order = c('CBWS_609', 'R_480', 'R_960', 'R_1440', 'R_1920',
                'R_2400', 'R_2880', 'R_3360', 'R_3840', 'R_4320',
                'D_480', 'D_960', 'D_1440', 'D_1920', 'D_2400',
                'D_2880', 'D_3360', 'D_3840', 'D_4320')

# need to replace dash - with underscore _, so the plot functions do not complain
data_df$hs_id <- gsub('-', '_', data_df$hs_id)

# H_null: the control (CBWS_609) is not better than other strategy. CBWS_609 <= OtherStrat. PMCMRplus format is OtherStrat - CBWS_609 >= 0
# H_alt: the control (CBWS_609) is better than other strategy. CBWS_609 > OtherStrat. 

data <- list()
i <- 1
for (task in task_info$task_col) {
  tmp_df <- data_df[data_df$task_col == task,]
  
  strategy_vector <- factor(tmp_df$hs_id, levels=hs_ids_order)
  total_hits_vector <- tmp_df$total_hits
  
  # perform KW test among strategies
  kw_test_res <- kruskalTest(x = total_hits_vector, g = strategy_vector)
  
  # kw-accept-alt: significant difference among strategies found.
  # conduct post-hoc conover-iman test with CBWS_609 as the control
  kwMOCI_test_res <- kwManyOneConoverTest(x = total_hits_vector, g = strategy_vector, 
                                          alternative="less", p.adjust.method = "holm")
  
  # makes a dataframe for the results
  kwMOCI_df <- toTidy(kwMOCI_test_res)
  kwMOCI_df$task_col = task
  
  # add SD (significant diff) or NSD (no-significant diff) for the kw test
  if (kw_test_res$p.value > 0.05){
    kwMOCI_df$kw_res = 'NSD'
  }
  else{
    kwMOCI_df$kw_res = 'SD'
  }
  
  # append to task test results to data
  data[[i]] <- kwMOCI_df
  
  i <- i + 1
}
  
# create one giant df with all task tests
kwMOCI_all_df <- rbindlist(data)

write.csv(kwMOCI_all_df, "./exp5_kwci_tests.csv", row.names=FALSE)
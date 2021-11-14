# fccml-thesis


# Experiment 1.x: Data processing, analysis, visualisation
- Builds the 3 graph datasets for two processing schemes: fixed size graph data sets and variable size datasets: ```fcc_experiment_1.x_both_fixed_and_variable_sized_graphs.ipynb```
- Builds the 3 graph datasets and saves graphs as torch objects 
```fcc_experiment_1_x_save_variable_graphs_as_torch_data_objects.ipynb```
- Visualisation of Node Metrics for Event ID:9 and Global Metrics of 100 Graphs 
```fccml_experiment_1_x_graph_metrics_for_100_events_and_node_metrics_for_event_id_9_visualisation.ipynb```


# Experiment 2.x GCN
## Network Depth 
- Model Training and Testing on GCN-2 on Fixed and Variable Sized Graphs and explanations of node predictions using GNNExplainer
```fcc_experiment_2_x_model_GCNX_2.ipynb```

- Model Training and Testing of GCN-16 on Variable Sized Graphs across 3 datasets
```fcc_experiment_2_x_model_final_GCNX_16_32_on_variable_sized.ipynb```

- Model Training and Testing of  On GCN-4 and GCN-8 on Variable Sized Graphs across 3 datasets
```fcc_experiment_2_x_model_final_GCNX_4_8_on_variable_sized.ipynb```


- Model Performance of GCN-16 across 3 datasets
```fcc_experiment_2_x_model_performance_GCNX_16.ipynb```

- Model Performance of GCN-4 across 3 datasets
```fcc_experiment_2_x_model_performance_GCNX_4.ipynb```

- Model Performance of GCN-8 across 3 datasets
```fcc_experiment_2_x_model_performance_GCNX_8.ipynb```

- Model Performance of  GCN-2 across 3 datasets
```fcc_experiment_2_x_model_performance_GCNX_2.ipynb```

- Model Performance by calculating the sum of 4 predicted properties and sum of 4 true properties of each of the clustered particles: H, Z and others separately in the test dataset and returning their MSE and MAE scores across each class for the 3 graph datasets created.
```fcc_experiment_2_x_model_GCNX_2_calculate_sum_of_mom_x_on_colab_save_physics_performance_metrics.ipynb```


## Hyperparameters
- Model Training and Testing on GCN-2 on the KNN dataset: Hyperparameter Dropout 
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_dropout_0_1_0_2_0_3_0_4_0_5.ipynb```

- Model Training and Testing on GCN-2 on the KNN dataset: Hidden Channels
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_hidden_channels_10_16_32_64_128.ipynb```

- Model Training and Testing on GCN-2 on the KNN dataset: Learning Rate 
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_lr_0_1_to_0_00001.ipynb```

- Model Training and Testing on GCN-2 on the KNN dataset: Non-linearity
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_non_linearity_elu_selu_gelu_leaky_relu_tanh.ipynb```

- Model Training and Testing on GCN-2 on the KNN dataset: Optimizers 
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_optimizer_adam_sgd_rmsprop.ipynb```

- Model Training and Testing on GCN-2 on the KNN dataset: Weight Decay 
```fcc_experiment_2_x_model_GCNX_2_hyperparameters_wd_0_5_to_0_00005.ipynb```

- Model Performance of GCN-2 dropout
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_dropout_0_1_0_2_0_3_0_4_0_5.ipynb```

- Model Performance of GCN-2 hidden channels
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_hidden_channels_10_16_32_64_128.ipynb```

- Model Performance of GCN-2 Learning Rate 
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_lr_0_1_to_0_00001.ipynb```

- Model Performance of GCN-2 Non Linearity 
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_non_linearity_elu_selu_gelu_leaky_relu_tanh.ipynb```

- Model Performance of GCN-2 optimizers
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_optimizer_adam_sgd_rmsprop.ipynb```

- Model Performance of GCN-2 Weight decay 
```fcc_experiment_2_x_model_performance_GCNX_2_hyperparameters_wd_0_5_to_0_00005.ipynb```

##  Dataset Size
- Model Training and Testing on GCN-2 on various dataset sizes:
```fcc_experiment_2_x_model_GCNX_2_on_20_40_60_80_percentage_of_dataset.ipynb```

- Model Performance of 20% of dataset
```fcc_experiment_2_x_model_performance_GCN_20_percent_dataset.ipynb```

- Model Performance of 40% of dataset
```fcc_experiment_2_x_model_performance_GCN_40_percent_dataset.ipynb```

- Model Performance of 60% of dataset 
```fcc_experiment_2_x_model_performance_GCN_60_percent_dataset.ipynb```

- Model Performance of 80% of dataset 
```fcc_experiment_2_x_model_performance_GCN_80_percent_dataset.ipynb```

# Experiment 3.x ChebNet
- Model Training and Testing of ChebNet-2 and ChebNet-4  on Variable Sized Graphs across 3 datasets
```fcc_experiment_3_x_model_ChebNet_2_4.ipynb```

- ChebNet-2 with k=3, 4 on all 3 datasets
```fcc_experiment_3_x_model_ChebNet_2_hyperparameters_k_3_4.ipynb```

- Model performance of ChebNet-2 on all 3 datasets 
```fcc_experiment_3_x_model_performance_ChebX_2.ipynb```

- Model performance ChebNet-2 with k=3, 4 on all 3 datasets 
```fcc_experiment_3_x_model_performance_ChebX_2_hyperparameters_k_3_4.ipynb```

- Model performance of ChebNet-4 on all 3 datasets 
```fcc_experiment_3_x_model_performance_ChebX_4.ipynb```


# Experiment 4.x SAGE
- Model Training and Testing of  SAGE-2 and SAGE-4  on Variable Sized Graphs across 3 datasets
```fcc_experiment_4_x_model_SAGE_2_4.ipynb```

- Model Performance of SAGE-2 across 3 datasets
```fcc_experiment_4_x_model_performance_SAGE_2.ipynb```

- Model Performance of SAGE-4 across 3 datasets
```fcc_experiment_4_x_model_performance_SAGE_4.ipynb```

# Experiment 5.x TAGCN
- Model Training and Testing of TAGCN-2 and TAGCN-4  on Variable Sized Graphs across 3 datasets
```fcc_experiment_5_x_model_TAGCN_2_4.ipynb```

- Model Performance TAGCN-2 on all datasets 
```fcc_experiment_5_x_model_performance_TAGCN_2.ipynb ```

- Model Performance of TAGCN-4 across all datasets 
```fcc_experiment_5_x_model_performance_TAGCN_4.ipynb```

# Experiment 6.x GAT
- Model Training and Testing of GAT-2 and GAT-4  on Variable Sized Graphs across 3 datasets
```fcc_experiment_6_x_model_GAT_2_4.ipynb```

- Model Training and Testing GAT-2 with different heads on KNN dataset
```fcc_experiment_6_x_model_GAT_2_hyperparameters_heads_2_4_6_8_10.ipynb```

- Model performance of GAT-2 across 3 datasets
```fcc_experiment_6_x_model_performance_GAT_2.ipynb```

- Model performance of GAT-2 with different heads on KNN dataset
```fcc_experiment_6_x_model_performance_GAT_2_hyperparameters_heads_2_4_6_8_10.ipynb```

- Model performance of GAT-4 across 3 datasets
```fcc_experiment_6_x_model_performance_GAT_4.ipynb```

# Experiment 7.x GIN
- Model Training and Testing of GIN-2 and GIN-4 on Variable Sized Graphs across 3 datasets
```fcc_experiment_7_x_model_GIN_2_4.ipynb ```

- Model performance of GIN-2 across 3 datasets
```fcc_experiment_7_x_model_performance_GIN_2.ipynb```

- Model performance GIN-4 across 3 datasets
```fcc_experiment_7_x_model_performance_GIN_4.ipynb```

# Experiment 8.x JK
- Model Training and Testing of JK-2 and JK-4 on Variable Sized Graphs across 3 datasets 
```fcc_experiment_8_x_model_JK_2_4.ipynb```

- Model Performance of JK-2 across 3 datasets 
```fcc_experiment_8_x_model_performance_JK_2.ipynb```

- Model Performance ofJK-4 across 3 datasets 
```fcc_experiment_8_x_model_performance_JK_4.ipynb```

# Experiment 9.x superGAT
- Model training and testing of superGAT-2 and superGAT-4 on all datasets
```fcc_experiment_9_x_model_superGAT_2_4.ipynb```

- Model training and testing of superGAT-2: different attentions and heads on KNN dataset
```fcc_experiment_9_x_model_superGAT_2_hyperparameters_attention_MX_SD_head_2_4_8.ipynb```

- Model Performance of superGAT-2 on all datasets
```fcc_experiment_9_x_model_performance_superGAT_2.ipynb```

- Model Performance of superGAT-2: different attentions and heads on KNN dataset
```fcc_experiment_9_x_model_performance_superGAT_2_hyperparameters_heads_2_4_6_8_10.ipynb```

- Model Performance of superGAT-4 on all datasets
```fcc_experiment_9_x_model_performance_superGAT_4.ipynb```


# Experiment 10.x MLP
- Model Training and Testing on MLP-2, MLP-4, MLP-8 on Variable Sized Graphs using only node features 
```fcc_experiment_10_x_model_MLP_2_4_8.ipynb```

- Model performance of MLP-2, MLP-4, MLP-8 on Variable Sized Graphs using event accuracy
```fcc_experiment_10_x_model_performance_MLP_2_4_8.ipynb```

# How to run each experiment 
- Each model training notebook is associated with a corresponding model performance notebook
## Model 
- Specify the path where the dataset is located. It could be directly uploaded in the session or could be hosted on your google drive. 
- If you use your google drive to host the dataset, mount your google drive and authenticate your account and connect your colab session with the drive ( just run the specific cell and it works). This is also useful if you want to save your model results into your google drive as well. Let this be Destination 1.
- If however you are just experimenting, uploading your dataset for the current session and storing the files temporarily then you can specify the path as '/content'. 
-  Specify the number of events to be used in the ML pipeline. If you want a quick check, keep the TOTAL_EVENTS = 10, if you want to run the complete pipeline then you can specify TOTAL_EVENTS = 10000. The upper limit is 10000 events so if you specify more than that, it might throw an error. 
## Model Performance
- If you didn't specify a google drive path, your output files of model training notebook are only limited to that session.
- For evaluating model performance specify the same Destination 1 (Google drive path) and run the notebook to get access to different measures of accuracy. 

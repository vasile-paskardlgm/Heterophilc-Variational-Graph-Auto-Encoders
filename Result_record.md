# (22/8/25) Note
    The GAE performs well(90%+)and stably when neg_edges are NOT uesd in testing|training at same time, BEST in NONE-NEGATIVE.

# (22/8/25) Note
    The VGAE performs well(90%+)when neg_edges are NONE in testing|validating, and neg_edges in training makes model stable.

# (22/8/26) Assumption
    The distribution in VGAE would be smooth, but the GT is sharp(0-1 matrix). Although we can predict a (w(i,j) ≠ 0), the ROC_AUC_CURVE would be low.

# (22/8/26) No training model with GCNConv in Pyg performs very good. Does it means the initialization method in Pyg or torch has huge contribution to solve the problem? What's more, the model.eval() is necessary if you want to get good performance WITHOUT TRAINING.

# (22/8/27) In NO TRAINING MODEL:
    The initialization method used in Pyg is glorot_uniform/kaiming_uniform, zero_start in batch_norm. It also performs well in all_kaiming_uniform. However, it performs very badly in using Gassian.
    The batch normalization is not needed.
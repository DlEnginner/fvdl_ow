from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis("/home/ftagalak/ray_results/GridSearchExperiment2")

# List all completed trials
print(analysis.trials)

# Get the best trial based on loss
best_trial = analysis.get_best_trial(metric="loss", mode="min")
print("Best trial config:", best_trial.config)
print("Best trial final loss:", best_trial.last_result["loss"])

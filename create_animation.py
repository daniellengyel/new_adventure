import new_adventure as na
import os

root_folder = os.environ["PATH_TO_ADV_FOLDER"]
data_name = "linear"
exp_name = "Newton_shift_est_IPM_Apr01_16-41-42_Daniels-MacBook-Pro-4.local"
experiment_folder = os.path.join(root_folder, "experiments", data_name, exp_name)

na.animations.get_exp_animations(experiment_folder)
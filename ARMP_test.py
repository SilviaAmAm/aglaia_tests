import numpy as np
from aglaia import aglaia
import joblib
import time

start_time = time.time()

# Test Data
input_data = "/Volumes/Transcend/repositories/Aglaia/data/local_slatm_qm7.npz"
# data = np.load(input_data)
#
# descriptors = data["arr_0"]
# zs = data["arr_1"]
# energies = data["arr_2"]

data = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_qm7_light.bz")
descriptors = data['descriptor']
zs = data["zs"]
energies = data["energies"]
energies_col = np.reshape(energies, (energies.shape[0], 1))

print("Loading the data took", str(time.time() - start_time), " to run")

# Setting up the estimator
estimator = aglaia.ARMP(iterations=1000, batch_size=10)

# Fitting to the data
estimator.fit(descriptors, zs, energies_col)

# Plotting the cost
estimator.plot_cost(filename="cost_test.png")

# Predicting
y_pred = estimator.predict([descriptors, zs])

# Plot correlation
estimator.correlation_plot(y_pred, energies, filename='correlation.png')



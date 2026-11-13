from dataset import normalize, denormalize, calculate_loss, modify_skf, test_md, trainset
from ml import generate_model, opimization_observation
from botorch.acquisition import (qExpectedImprovement, qNoisyExpectedImprovement)


train_x, train_obj = trainset(n_samples=10, lower=-0.6, upper=0.0)
mll, model = generate_model(train_x, train_obj)
new_x_set, new_obj_set = opimization_observation(qLogExpectedImprovement(model=model, best_f=train_obj.max()), model, train_x, train_obj)
mll_updated, model_updated = generate_model(new_x_set, new_obj_set)

for i in range(10):
    new_x, new_obj = opimization_observation(qLogExpectedImprovement(model=model_new, best_f=train_obj.max()), model, new_x, new_obj)
    mll_new, model_new = generate_model(new_x, new_obj)

print('Optimized Parameters : ', denormalize(new_x))
print('Optimized RMSE : ', -new_obj)

# To compare with initial data
force, energy=test_md('../trans3d-0-1/')
print('Force RMSE : ', force)
print('Energz RMSE : ', energy)
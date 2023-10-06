
#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
from src.device import device_info
from src.data import GeoDataset
from src.model import GCN_Geo 
from src.process import train, validation, predict_test
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from math import sqrt

device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
dataset = GeoDataset(root='data')
print('Number of NODES features: ', dataset.num_features)
print('Number of EDGES features: ', dataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - finish_time_preprocessing) / 60 #TODO


# # Number of datapoints in the training set:
training_test_percentage = 0.8
n_train = int(len(dataset) * training_test_percentage)

# # Number of datapoints in the validation set:
n_val = len(dataset) - n_train

# # Define pytorch training and validation set objects:
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# # Build pytorch training and validation set dataloaders:
batch_size = 25
dataloader = DataLoader(dataset, batch_size, shuffle=True)
train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)

## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Set up model:
# Initial Inputs
initial_dim_gcn = dataset.num_features
edge_dim_feature = dataset.num_edge_features

hidden_dim_nn_1 = 500
p1 = 0
hidden_dim_nn_2 = 250    
p2 = 0
hidden_dim_nn_3 = 100
p3 = 0

hidden_dim_gat_0 = 10
hidden_dim_gat_1 = 0

hidden_dim_fcn_1 = 100
hidden_dim_fcn_2 = 50
hidden_dim_fcn_3 = 10


model = GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                p1,
                hidden_dim_nn_2,
                p2,
                hidden_dim_nn_3,
                p3,
                hidden_dim_gat_0,
                hidden_dim_gat_1,
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3
            ).to(device)


# Set up optimizer:
learning_rate = 1E-3
optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 100
for epoch in range(1, number_of_epochs):
    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), "best_model_weights.pth")
     
finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60

#Testing:
weights_file = "best_model_weights.pth"

# Training:
input_all, target_all_train, pred_prob_all_train = predict_test(model, train_dataloader, device, weights_file)


r2_train = r2_score(target_all_train.cpu(), pred_prob_all_train.cpu())
r_train, _ = pearsonr(target_all_train.cpu(), pred_prob_all_train.cpu()) 
mae_train = mean_absolute_error(target_all_train.cpu(), pred_prob_all_train.cpu())
mse_train = mean_squared_error(target_all_train.cpu(), pred_prob_all_train.cpu(), squared=False)
rmse_train = sqrt(mse_train)

# Validation:
input_all, target_all_test, pred_prob_all_test = predict_test(model, val_dataloader, device,weights_file)


r2_validation = r2_score(target_all_test.cpu(), pred_prob_all_test.cpu())
r_validation, _ = pearsonr(target_all_test.cpu(), pred_prob_all_test.cpu())
mae_validation = mean_absolute_error(target_all_test.cpu(), pred_prob_all_test.cpu())
mse_validation = mean_squared_error(target_all_test.cpu(), pred_prob_all_test.cpu(), squared=False)
rmse_validation = sqrt(mse_validation)



dataset_path = "data/unmod.csv"
df = pd.read_csv(dataset_path)
RT_values = df.iloc[:, 1]  
#Plots
#Distribution Dataset
plt.hist(RT_values, bins=10)  
plt.title("RT Distribution-unmod.txt")
plt.xlabel("RT Values (sec)")
plt.ylabel("Frequency")
plt.show()

#Lose curves
plt.plot(train_losses, label='Train losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Train losses')

plt.plot(val_losses, label='Validation losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Losses')
plt.savefig('results/histogram_dataset', format="png")
plt.savefig('results/losses', format="jpg")
plt.show()


#Training

legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nMSE: {:.4f}\nRMSE: {:.4f}".format(
    r2_train, r_train, mae_train, mse_train, rmse_train
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_train.cpu(), pred_prob_all_train.cpu(), alpha=0.3)
plt.plot([min(target_all_train.cpu()), max(target_all_train.cpu())], [min(target_all_train.cpu()), max(target_all_train.cpu())], color="k", ls="--")
plt.xlim([min(target_all_train.cpu()), max(target_all_train.cpu())])
plt.title('Training')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.savefig('results/training', format="jpg")
plt.show()

#Validation
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nMSE: {:.4f}\nRMSE: {:.4f}".format(
    r2_validation, r_validation, mae_validation, mse_validation, rmse_validation
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_test.cpu(), pred_prob_all_test.cpu(), alpha=0.3)
plt.plot([min(target_all_test.cpu()), max(target_all_test.cpu())], [min(target_all_test.cpu()), max(target_all_test.cpu())], color="k", ls="--")
plt.xlim([min(target_all_test.cpu()), max(target_all_test.cpu())])
plt.title('Validation')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.savefig('results/test', format="jpg")
plt.show()


#Times
finish_time = time.time()
time_prediction = (finish_time - finish_time_training) / 60
total_time = (finish_time - start_time) / 60
print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
print("\n //// Training time: {:3f} minutes ////".format(time_training))
print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
print("\n //// Total time: {:3f} minutes ////".format(total_time))

# Result DataFrame
data = {
    "Metric": [
        "number_features",
        "num_edge_features",
        "initial_dim_gcn ",
        "edge_dim_feature",
        "hidden_dim_nn_1 ",
        "p1 ",
        "hidden_dim_nn_2 ",
        "p2 ",
        "hidden_dim_nn_3 ",
        "p3 ",
        "hidden_dim_gat_0",
        "hidden_dim_gat_1",
        "hidden_dim_fcn_1 ",
        "hidden_dim_fcn_2 ",
        "hidden_dim_fcn_3 ",
        "training_test_percentage %",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "r2_train",
        "r_train",
        "mae_train",
        "mse_train", 
        "rmse_train", 
        "r2_validation",
        "r_validation",
        "mae_validation",
        "mse_validation",
        "rmse_validation",
        "time_preprocessing", 
        "time_training",
        "time_prediction",
        "total_time"
    ],
    "Value": [
        dataset.num_features,
        dataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        p1 ,
        hidden_dim_nn_2 ,
        p2 ,
        hidden_dim_nn_3,
        p3 ,
        hidden_dim_gat_0,
        hidden_dim_gat_1,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        training_test_percentage*100,
        batch_size,
        learning_rate,
        number_of_epochs,
        r2_train, 
        r_train, 
        mae_train, 
        mse_train,
        rmse_train,
        r2_validation,
        r_validation,
        mae_validation, 
        mse_validation,
        rmse_validation,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}

df = pd.DataFrame(data)
df.to_csv('results/results.csv', index=False)



# %%

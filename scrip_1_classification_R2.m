clear all;
close all;
clc;
addpath('../');


%%%%%%%%%%%%%%%%%%%%% data base %%%%%%%%%%%%%%%%%%%%
m = 50;
x1 = [1/2 + 0.15*randn(1,m),-1/2 + 0.15*randn(1,m), 0.15*randn(1,m)]
x2 = [1/2 + 0.15*randn(1,m), 0.15*randn(1,m), -1/2 + 0.15*randn(1,m)];
y1 = [ones(1,m),zeros(1,m),zeros(1,m)];
y2 = [zeros(1,m),ones(1,m),zeros(1,m)];
y3 = [zeros(1,m),zeros(1,m),ones(1,m)];

database.X_train = [x1;x2];
database.Y_train = [y1;y2;y3];




%-- Display training database
visu.display_point_cloud(database,'Training dataset');


%-- Build a model with a n_h-dimensional hidden layer
num_iterations = 20000;
learning_rate = 0.1;
print_cost = true;
nX = size(database.X_train,1);
layers_dims = [nX,2,3];
[parameters,costs] = L_layers_nn.model(database, layers_dims, num_iterations, learning_rate, print_cost);


%-- Display decision boundary
visu.display_decision_boundary(database,parameters);


%-- Compute accuracy
X_train = database.X_train;
Y_train = database.Y_train;
Y_prediction_train = L_layers_nn.predict(parameters, X_train);



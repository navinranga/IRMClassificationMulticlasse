clear all;
close all;
clc;
addpath('../');

%-- Load database parameters
filename = './data/data_Clanu_22.mat';
load(filename);

%-- Affichage image
%img1=T1_C{1};
%figure; imagesc(img1); axis image; colormap(gray); colorbar;

nC = 9;
%-- put database in place
[database] = database.structure_database(T1_A,T1_C,T1_S,T2_A,T2_C,T2_S,PD_A,PD_C,PD_S,nC);

%-- Display training database
%visu.display_clanu_database(database);

%paramètres optimaux pour nC=9 :
%iter 3000 learn 0.01 layersdim 10 10

%-- Build a model with a n_h-dimensional hidden layer
num_iterations =3000;
learning_rate = 0.01;
print_cost = true;
nX = size(database.X_train,1);

layers_dims = [nX,10,10,nC];
[parameters,costs] = L_layers_nn.model(database, layers_dims, num_iterations, learning_rate, print_cost);


%-- Compute accuracy
X_train = database.X_train;
Y_train = database.Y_train;
X_valid = database.X_valid;
Y_valid = database.Y_valid;
X_test = database.X_test;
Y_test = database.Y_test;


nB =  size(database.Y_train,2);
Y_train_c = sum(([1:nC]'*ones(1,nB)).*Y_train,1); 
nB =  size(database.Y_test,2);
Y_test_c = sum(([1:nC]'*ones(1,nB)).*Y_test,1);
nB =  size(database.Y_valid,2);
Y_valid_c = sum(([1:nC]'*ones(1,nB)).*Y_valid,1);

Y_prediction_train = L_layers_nn.predict(parameters, X_train);
Y_prediction_valid = L_layers_nn.predict(parameters, X_valid);
Y_prediction_test = L_layers_nn.predict(parameters, X_test);


%-- Print train/test Errors
disp(['train accuracy: ', num2str(100 - mean(sum(abs(Y_prediction_train - Y_train_c),1)) * 100), ' %'])
disp(['valid accuracy: ', num2str(100 - mean(sum(abs(Y_prediction_valid - Y_valid_c),1)) * 100), ' %'])
disp(['test accuracy: ', num2str(100 - mean(sum(abs(Y_prediction_test - Y_test_c),1)) *100)  , ' %'])    


clear all
clc
[Data] = xlsread('data.xlsx')

cov_matrix = cov(Data);

% Compute eigenvectors and eigenvalues
[eig_vec, eig_val] = eig(cov_matrix);

% Sort eigenvectors based on eigenvalues
[~, idx] = sort(diag(eig_val), 'descend');
eig_vec_sorted = eig_vec(:, idx);
idx;
% Project data onto principal components
FinalDataSet_Ferm = Data * eig_vec_sorted;

size(FinalDataSet_Ferm)
%Data_train = zeros(250, 3); % Initialize Data_train to store 500 rows and 3 columns

eig_val = sort(eig_val,'descend')
eigenvalues = eig_val(1,:)
eigenvalues = sort(eigenvalues,'descend')
Per_eig_val = eigenvalues./(sum(eigenvalues))

j = 7;      % No. of Features selected
for i=1:1:1180
    Data_train_Ferm(i,:)=FinalDataSet_Ferm(i,1:j);
end
for i=1:1:1180
    Data_test_Ferm(i,:)=FinalDataSet_Ferm(1180+i,1:j); 
end

autoenc_Ferm = trainAutoencoder(Data_train_Ferm,'MaxEpochs',4000)

%{
L2 = 0.004;
SR = 6;
SP = 0.15;

hiddenSize=15;
autoenc_Ferm = trainAutoencoder(Data_train_Ferm,hiddenSize, ...
'EncoderTransferFunction','satlin', ...
'DecoderTransferFunction','satlin', ...
'MaxEpochs',2000, ...
'L2WeightRegularization',L2, ...   
'SparsityRegularization',SR, ...
'SparsityProportion',SP, ...
'ScaleData',false);
%}
xReconstructed_Ferm = predict(autoenc_Ferm,Data_test_Ferm)
mseError_Ferm = mse(Data_test_Ferm-xReconstructed_Ferm)

figure('Name','True State vs Reconstructed True State (PCA)')
subplot(411)
plot(1181:2360,FinalDataSet_Ferm(1181:2360,1),'b',1181:2360,xReconstructed_Ferm(:,1),'r')
xlabel('Sampling Instant')
ylabel('Feature 1')
legend('True State', 'Reconstructed True State')

subplot(412)
plot(1181:2360,FinalDataSet_Ferm(1181:2360,2),'b',1181:2360,xReconstructed_Ferm(:,2),'r')
xlabel('Sampling Instant')
ylabel('Feature 2')
legend('True State', 'Reconstructed True State')

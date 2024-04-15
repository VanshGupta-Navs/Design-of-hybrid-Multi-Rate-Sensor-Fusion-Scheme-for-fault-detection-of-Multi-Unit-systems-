clear all
clc
load result_openloop_simulation.mat

% Without PCA
for i=1:1:250
err_normal(i,:) = xReconstructed(i,:) - x_test(i,:)
err_abnormal_abnormal(i,:) = NoisyxReconstructed(i,:) - x_Noisy1(i,:)
err_abnormal_normal(i,:) = NoisyxReconstructed(i,:) - x_test(i,:)
end

figure('Name','Error Plot bet. Normal and Noisy Data 1')
subplot(411)
plot(251:500,err_normal(:,1),'b',251:500,err_abnormal_abnormal(:,1),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data (Ca)')
legend('Normal Error','Noisy Error')

subplot(412)
plot(251:500,err_normal(:,2),'b',251:500,err_abnormal_abnormal(:,2),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data (T)')
legend('Normal Error','Noisy Error')

subplot(413)
plot(251:500,err_normal(:,3),'b',251:500,err_abnormal_abnormal(:,3),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(u1_flow)')
legend('Normal Error','Noisy Error')

subplot(414)
plot(251:500,err_normal(:,4),'b',251:500,err_abnormal_abnormal(:,4),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(u2_temp)')
legend('Normal Error','Noisy Error')

%--------------------------------------------------------------------------

figure('Name','Error Plot bet. Normal and Noisy Data 2')
subplot(411)
plot(251:500,err_normal(:,1),'b',251:500,err_abnormal_normal(:,1),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(Ca)')
legend('Normal Error','Noisy Error')

subplot(412)
plot(251:500,err_normal(:,2),'b',251:500,err_abnormal_normal(:,2),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(T)')
legend('Normal Error','Noisy Error')

subplot(413)
plot(251:500,err_normal(:,3),'b',251:500,err_abnormal_normal(:,3),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(u1_flow)')
legend('Normal Error','Noisy Error')

subplot(414)
plot(251:500,err_normal(:,4),'b',251:500,err_abnormal_normal(:,4),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed data(u2_temp)')
legend('Normal Error','Noisy Error')

%--------------------------------------------------------------------------
% With PCA
for i = 1:1:250
err_pca(i,:) = xReconstructed2(i,:) - FinalDataSet(250+i,1:2);
err_pca_Noisy(i,:) = NoisyxReconstructed_PCA(i,:) - FinalDataSet_Noisy(i,1:2)
end
figure('Name','Error Plot bet. Normal and Noisy Features(PCA)')
subplot(211)
plot(1:250,err_pca(:,1),'b ',1:250,err_pca_Noisy(:,1),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed Feature 1')
legend('Normal Error','Noisy Error')
subplot(212)
plot(1:250,err_pca(:,2),'b ',1:250,err_pca_Noisy(:,2),'r')
xlabel('Sampling Instant')
ylabel('Error bet. Input and Reconstructed Feature 2')
legend('Normal Error','Noisy Error')


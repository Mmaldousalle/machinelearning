clc
clear all
close all

%% =========================================================
% Load data
%% =========================================================
load('data_D2_C3_S1.mat')

X_original = double(p.valor);      % Features: 2 x N
Y = p.clase(:);                    % Labels: N x 1

[D,N] = size(X_original);

fprintf('\nDataset loaded correctly\n');
fprintf('Number of features: %d\n', D);
fprintf('Number of samples: %d\n', N);

%% =========================================================
% Train/Test split 70/30
%% =========================================================
rng(0)

cv = cvpartition(Y,'HoldOut',0.30);

idxTrain = training(cv);
idxTest  = test(cv);

X_train_raw = X_original(:,idxTrain);
Y_train = Y(idxTrain);

X_test_raw = X_original(:,idxTest);
Y_test = Y(idxTest);

%% =========================================================
% Plot original data
%% =========================================================
figure
gscatter(X_original(1,:),X_original(2,:),Y)
xlabel('Feature 1')
ylabel('Feature 2')
title('Original Synthetic Dataset')
grid on

%% =========================================================
% Normalization using only training data
%% =========================================================
meant = mean(X_train_raw,2);
stdt = std(X_train_raw,0,2);
stdt(stdt==0) = 1;

X_train_norm = (X_train_raw - meant)./stdt;
X_test_norm  = (X_test_raw  - meant)./stdt;

X_train_norm(isnan(X_train_norm)) = 0;
X_test_norm(isnan(X_test_norm)) = 0;

%% =========================================================
% PCA using only training data
%% =========================================================
[coeff,~,latent] = pca(X_train_norm');

explainedVariance = latent / sum(latent);

fprintf('\n--- PCA Explained Variance ---\n');
disp(explainedVariance)

componentes = 1:D;

error_train_pca = zeros(length(componentes),1);
error_test_pca  = zeros(length(componentes),1);
error_recon_pca = zeros(length(componentes),1);

%% =========================================================
% PCA + SVM loop for PCA component evaluation
%% =========================================================
for c = 1:length(componentes)

    nPCA_temp = componentes(c);

    transMat_temp = coeff(:,1:nPCA_temp)';

    ProyTrain_temp = transMat_temp * X_train_norm;
    ProyTest_temp  = transMat_temp * X_test_norm;

    X_train_temp = ProyTrain_temp';
    X_test_temp  = ProyTest_temp';

    mu_temp = mean(X_train_temp);
    sigma_temp = std(X_train_temp);
    sigma_temp(sigma_temp==0) = 1;

    X_train_temp_norm = (X_train_temp - mu_temp)./sigma_temp;
    X_test_temp_norm  = (X_test_temp  - mu_temp)./sigma_temp;

    t_temp = templateSVM( ...
        'KernelFunction','gaussian', ...
        'KernelScale','auto', ...
        'BoxConstraint',1, ...
        'Standardize',false);

    svm_temp = fitcecoc( ...
        X_train_temp_norm, ...
        Y_train, ...
        'Learners',t_temp);

    Y_pred_train_temp = predict(svm_temp,X_train_temp_norm);
    Y_pred_test_temp  = predict(svm_temp,X_test_temp_norm);

    error_train_pca(c) = mean(Y_pred_train_temp ~= Y_train);
    error_test_pca(c)  = mean(Y_pred_test_temp ~= Y_test);

    X_recon_norm_temp = transMat_temp' * ProyTrain_temp;
    error_recon_pca(c) = mse(X_train_norm - X_recon_norm_temp);

    fprintf('nPCA=%d | Train Error=%.4f | Test Error=%.4f | Recon Error=%.4f\n', ...
        nPCA_temp, ...
        error_train_pca(c), ...
        error_test_pca(c), ...
        error_recon_pca(c));
end

%% =========================================================
% PCA component selection
%% =========================================================
[~,idxBestPCA] = min(error_test_pca);
nPCA = componentes(idxBestPCA);

fprintf('\n--- Selected PCA components ---\n');
fprintf('Selected nPCA: %d\n', nPCA);
fprintf('Minimum test error: %.4f\n', error_test_pca(idxBestPCA));
fprintf('Associated reconstruction error: %.4f\n', error_recon_pca(idxBestPCA));

%% =========================================================
% PCA error graph
%% =========================================================
figure

plot(componentes,error_train_pca,'-o','LineWidth',2)
hold on
plot(componentes,error_test_pca,'-s','LineWidth',2)
plot(componentes,error_recon_pca,'-^','LineWidth',2)

xlabel('Number of PCA Components')
ylabel('Error')
title('PCA Error Behavior - Synthetic Data + SVM')

legend('Training Error','Test Error','Reconstruction Error')

grid on
plot(nPCA,error_test_pca(idxBestPCA),'r*','MarkerSize',12)
hold off

%% =========================================================
% Final PCA projection
%% =========================================================
transMat = coeff(:,1:nPCA)';

ProyTrain = transMat * X_train_norm;
ProyTest  = transMat * X_test_norm;

X_train = ProyTrain';
X_test  = ProyTest';

%% =========================================================
% Final reconstruction error
%% =========================================================
X_recon_train_norm = transMat' * ProyTrain;

Emsen = mse(X_train_norm - X_recon_train_norm);

fprintf('\n--- Final PCA reconstruction error ---\n');
fprintf('Normalized reconstruction MSE: %.6f\n', Emsen);

%% =========================================================
% Normalize PCA components using training data
%% =========================================================
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma==0) = 1;

X_train_svm = (X_train - mu)./sigma;
X_test_svm  = (X_test  - mu)./sigma;

%% =========================================================
% Iterative SVM training using fitcecoc
%% =========================================================
boxValues = [0.1 1 10 100];
kernelScaleValues = [0.1 1 10];

results = [];
bestAccuracy = 0;

for i = 1:length(boxValues)
    for j = 1:length(kernelScaleValues)

        t = templateSVM( ...
            'KernelFunction','gaussian', ...
            'KernelScale',kernelScaleValues(j), ...
            'BoxConstraint',boxValues(i), ...
            'Standardize',false);

        svm_model_temp = fitcecoc( ...
            X_train_svm, ...
            Y_train, ...
            'Learners',t);

        Y_pred_train_temp = predict(svm_model_temp,X_train_svm);
        Y_pred_test_temp  = predict(svm_model_temp,X_test_svm);

        trainError = mean(Y_pred_train_temp ~= Y_train);
        testError  = mean(Y_pred_test_temp ~= Y_test);

        acc = (1 - testError)*100;

        results = [results; ...
            boxValues(i), ...
            kernelScaleValues(j), ...
            trainError, ...
            testError, ...
            acc];

        if acc > bestAccuracy
            bestAccuracy = acc;
            svm_model = svm_model_temp;
            bestPred = Y_pred_test_temp;
            bestBox = boxValues(i);
            bestKernelScale = kernelScaleValues(j);
        end
    end
end

%% =========================================================
% SVM results table
%% =========================================================
results_table = array2table(results, ...
    'VariableNames', ...
    {'BoxConstraint','KernelScale','TrainError','TestError','Accuracy'})

fprintf('\n--- Best SVM Configuration ---\n');
fprintf('Best Accuracy: %.2f%%\n', bestAccuracy);
fprintf('Best BoxConstraint: %.2f\n', bestBox);
fprintf('Best KernelScale: %.2f\n', bestKernelScale);

%% =========================================================
% Final SVM evaluation
%% =========================================================
Y_pred = bestPred;
accuracy = bestAccuracy;

fprintf('\n--- Final Results PCA + SVM ---\n');
fprintf('Selected PCA components: %d\n', nPCA);
fprintf('Test Accuracy: %.2f%%\n', accuracy);

%% =========================================================
% Confusion matrix
%% =========================================================
figure

confusionchart(Y_test,Y_pred, ...
    'Title','Confusion Matrix - PCA + SVM Synthetic Data', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% =========================================================
% Accuracy bar graph
%% =========================================================
figure

bar(results_table.Accuracy)

set(gca,'XTick',1:height(results_table))

xticklabels(strcat("C=", string(results_table.BoxConstraint), ...
                   " / KS=", string(results_table.KernelScale)))

xtickangle(45)

ylabel('Accuracy (%)')
xlabel('SVM Configuration')
title('Iterative Results - PCA + SVM Synthetic Data')

grid on

[maxAcc, idxBest] = max(results_table.Accuracy);

hold on
plot(idxBest,maxAcc,'r*','MarkerSize',12)
hold off

%% =========================================================
% Train vs Test Error graph
%% =========================================================
figure

plot(results_table.TrainError,'-o','LineWidth',1.5)
hold on
plot(results_table.TestError,'-o','LineWidth',1.5)

set(gca,'XTick',1:height(results_table))

xticklabels(strcat("C=", string(results_table.BoxConstraint), ...
                   " / KS=", string(results_table.KernelScale)))

xtickangle(45)

ylabel('Error')
xlabel('SVM Configuration')
title('Training vs Test Error - PCA + SVM Synthetic Data')

legend('Train Error','Test Error')

grid on
hold off

%% =========================================================
% Decision boundary plot
%% =========================================================
if nPCA == 2

    figure

    x1range = linspace(min(X_train_svm(:,1))-1,max(X_train_svm(:,1))+1,150);
    x2range = linspace(min(X_train_svm(:,2))-1,max(X_train_svm(:,2))+1,150);

    [x1Grid,x2Grid] = meshgrid(x1range,x2range);

    X_grid = [x1Grid(:), x2Grid(:)];

    Y_grid = predict(svm_model,X_grid);

    gscatter(X_train_svm(:,1),X_train_svm(:,2),Y_train)
    hold on

    contour(x1Grid,x2Grid,reshape(double(Y_grid),size(x1Grid)), ...
        'LineWidth',1.5)

    xlabel('PCA Component 1')
    ylabel('PCA Component 2')
    title('Decision Boundary - PCA + SVM Synthetic Data')

    grid on
    hold off

else

    figure

    gscatter(X_train_svm(:,1),zeros(size(X_train_svm(:,1))),Y_train)

    xlabel('PCA Component 1')
    ylabel('Reference Axis')
    title('Training Data Projection - PCA + SVM Synthetic Data')

    grid on

end

%% =========================================================
% Random individual prediction
%% =========================================================
rng('shuffle')

ind_random = randi(length(Y_test));

x_new = X_test_svm(ind_random,:);
real_label = Y_test(ind_random);

pred_label = predict(svm_model,x_new);

fprintf('\n--- Random Individual Prediction ---\n');
fprintf('Real class: %d\n', real_label);
fprintf('Predicted class: %d\n', pred_label);

figure

if nPCA == 2
    scatter(X_test_svm(:,1),X_test_svm(:,2),30,Y_test,'filled')
    hold on
    scatter(x_new(1),x_new(2),150,'r','filled')

    xlabel('PCA Component 1')
    ylabel('PCA Component 2')
else
    scatter(X_test_svm(:,1),zeros(size(X_test_svm(:,1))),30,Y_test,'filled')
    hold on
    scatter(x_new(1),0,150,'r','filled')

    xlabel('PCA Component 1')
    ylabel('Reference Axis')
end

title(['SVM predicts: ',num2str(pred_label), ...
       ' | Real: ',num2str(real_label)])

grid on
hold off
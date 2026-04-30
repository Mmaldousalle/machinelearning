clc
clear all
close all

%% =========================================================
% Load data
%% =========================================================
load('data_D2_C3_S1.mat')

X_original = double(p.valor);      % Features: 2 x N
Y_original = p.clase(:);           % Original labels

[D,N] = size(X_original);

[classLabels,~,Y_idx] = unique(Y_original,'stable');
numClasses = length(classLabels);

fprintf('\nDataset loaded correctly\n');
fprintf('Number of features: %d\n', D);
fprintf('Number of samples: %d\n', N);
fprintf('Number of classes: %d\n', numClasses);

%% =========================================================
% Train/Test split 70/30
%% =========================================================
rng(0)

cv = cvpartition(Y_idx,'HoldOut',0.30);

idxTrain = training(cv);
idxTest  = test(cv);

X_train_raw = X_original(:,idxTrain);
Y_train_idx = Y_idx(idxTrain);
Y_train_original = Y_original(idxTrain);

X_test_raw = X_original(:,idxTest);
Y_test_idx = Y_idx(idxTest);
Y_test_original = Y_original(idxTest);

%% =========================================================
% Plot original data
%% =========================================================
figure
gscatter(X_original(1,:),X_original(2,:),Y_original)
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

Y_train_onehot = full(ind2vec(Y_train_idx'));

%% =========================================================
% PCA + ANN loop for PCA component selection
%% =========================================================
for c = 1:length(componentes)

    nPCA_temp = componentes(c);

    transMat_temp = coeff(:,1:nPCA_temp)';

    ProyTrain_temp = transMat_temp * X_train_norm;
    ProyTest_temp  = transMat_temp * X_test_norm;

    X_recon_train_norm = transMat_temp' * ProyTrain_temp;
    error_recon_pca(c) = mse(X_train_norm - X_recon_train_norm);

    X_train_temp = ProyTrain_temp';
    X_test_temp  = ProyTest_temp';

    mu_temp = mean(X_train_temp);
    sigma_temp = std(X_train_temp);
    sigma_temp(sigma_temp==0) = 1;

    X_train_ann_temp = (X_train_temp - mu_temp)./sigma_temp;
    X_test_ann_temp  = (X_test_temp  - mu_temp)./sigma_temp;

    hiddenLayerSize = 20;

    net_temp = patternnet(hiddenLayerSize);

    net_temp.trainFcn = 'trainscg';
    net_temp.performFcn = 'crossentropy';
    net_temp.divideFcn = 'dividetrain';
    net_temp.trainParam.epochs = 300;
    net_temp.trainParam.showWindow = false;

    net_temp = train(net_temp,X_train_ann_temp',Y_train_onehot);

    Y_train_prob = net_temp(X_train_ann_temp');
    [~,Y_train_pred_idx] = max(Y_train_prob,[],1);
    Y_train_pred_idx = Y_train_pred_idx';

    Y_test_prob = net_temp(X_test_ann_temp');
    [~,Y_test_pred_idx] = max(Y_test_prob,[],1);
    Y_test_pred_idx = Y_test_pred_idx';

    error_train_pca(c) = mean(Y_train_pred_idx ~= Y_train_idx);
    error_test_pca(c)  = mean(Y_test_pred_idx ~= Y_test_idx);

    fprintf('nPCA=%d | Train Error=%.4f | Test Error=%.4f | Recon Error=%.4f\n', ...
        nPCA_temp, ...
        error_train_pca(c), ...
        error_test_pca(c), ...
        error_recon_pca(c));

end

%% =========================================================
% Select best PCA number
%% =========================================================
[~,idxBestPCA] = min(error_test_pca);

nPCA = componentes(idxBestPCA);

fprintf('\n--- Selected PCA Components for ANN ---\n');
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
title('PCA Error Behavior - Synthetic Data + ANN')

legend('Training Error ANN','Test Error ANN','Reconstruction Error PCA')

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
% Normalize PCA components
%% =========================================================
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma==0) = 1;

X_train_ann = (X_train - mu)./sigma;
X_test_ann  = (X_test  - mu)./sigma;

Y_train_onehot = full(ind2vec(Y_train_idx'));

%% =========================================================
% Iterative ANN training
%% =========================================================
hiddenValues = [10 20 30 40];
epochValues = [100 300];

resultsANN = [];
bestAccuracy = 0;

for i = 1:length(hiddenValues)
    for j = 1:length(epochValues)

        hiddenLayerSize = hiddenValues(i);

        net_temp = patternnet(hiddenLayerSize);

        net_temp.trainFcn = 'trainscg';
        net_temp.performFcn = 'crossentropy';
        net_temp.divideFcn = 'dividetrain';

        net_temp.trainParam.epochs = epochValues(j);
        net_temp.trainParam.goal = 1e-6;
        net_temp.trainParam.showWindow = false;

        [net_temp,tr_temp] = train(net_temp,X_train_ann',Y_train_onehot);

        Y_pred_train_prob = net_temp(X_train_ann');
        [~,Y_pred_train_idx] = max(Y_pred_train_prob,[],1);
        Y_pred_train_idx = Y_pred_train_idx';

        trainError = mean(Y_pred_train_idx ~= Y_train_idx);

        Y_pred_test_prob = net_temp(X_test_ann');
        [~,Y_pred_test_idx] = max(Y_pred_test_prob,[],1);
        Y_pred_test_idx = Y_pred_test_idx';

        testError = mean(Y_pred_test_idx ~= Y_test_idx);
        accuracy = (1 - testError)*100;

        resultsANN = [resultsANN; ...
            hiddenLayerSize, ...
            epochValues(j), ...
            trainError, ...
            testError, ...
            accuracy];

        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            net = net_temp;
            tr = tr_temp;
            bestPredIdxANN = Y_pred_test_idx;
            bestHiddenLayerSize = hiddenLayerSize;
            bestEpochs = epochValues(j);
        end

    end
end

%% =========================================================
% ANN results table
%% =========================================================
resultsANN_table = array2table(resultsANN, ...
    'VariableNames', ...
    {'HiddenNeurons','Epochs','TrainError','TestError','Accuracy'})

fprintf('\n--- Best ANN Configuration ---\n');
fprintf('Best Accuracy ANN: %.2f%%\n', bestAccuracy);
fprintf('Best hidden neurons: %d\n', bestHiddenLayerSize);
fprintf('Best epochs: %d\n', bestEpochs);

%% =========================================================
% Final ANN evaluation
%% =========================================================
Y_pred_idx = bestPredIdxANN;
Y_pred_original = classLabels(Y_pred_idx);

accuracy = bestAccuracy;

fprintf('\n--- Final Results PCA + ANN ---\n');
fprintf('Selected PCA components: %d\n', nPCA);
fprintf('Hidden neurons: %d\n', bestHiddenLayerSize);
fprintf('Epochs: %d\n', bestEpochs);
fprintf('Test Accuracy ANN: %.2f%%\n', accuracy);

%% =========================================================
% Confusion matrix
%% =========================================================
figure

confusionchart(Y_test_original,Y_pred_original, ...
    'Title','Confusion Matrix - PCA + ANN Synthetic Data', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% =========================================================
% Accuracy bar graph
%% =========================================================
figure

bar(resultsANN_table.Accuracy)

set(gca,'XTick',1:height(resultsANN_table))

xticklabels(strcat("N=", string(resultsANN_table.HiddenNeurons), ...
                   " / E=", string(resultsANN_table.Epochs)))

xtickangle(45)

ylabel('Accuracy (%)')
xlabel('ANN Configuration')
title('Iterative Results - PCA + ANN Synthetic Data')

grid on

[maxAcc, idxBest] = max(resultsANN_table.Accuracy);

hold on
plot(idxBest,maxAcc,'r*','MarkerSize',12)
hold off

%% =========================================================
% Train vs Test Error graph
%% =========================================================
figure

plot(resultsANN_table.TrainError,'-o','LineWidth',1.5)
hold on
plot(resultsANN_table.TestError,'-o','LineWidth',1.5)

set(gca,'XTick',1:height(resultsANN_table))

xticklabels(strcat("N=", string(resultsANN_table.HiddenNeurons), ...
                   " / E=", string(resultsANN_table.Epochs)))

xtickangle(45)

ylabel('Error')
xlabel('ANN Configuration')
title('Training vs Test Error - PCA + ANN Synthetic Data')

legend('Train Error','Test Error')

grid on
hold off

%% =========================================================
% Decision region / projection plot
%% =========================================================
if nPCA == 2

    figure

    x1range = linspace(min(X_train_ann(:,1))-1,max(X_train_ann(:,1))+1,150);
    x2range = linspace(min(X_train_ann(:,2))-1,max(X_train_ann(:,2))+1,150);

    [x1Grid,x2Grid] = meshgrid(x1range,x2range);

    X_grid = [x1Grid(:), x2Grid(:)];

    Y_grid_prob = net(X_grid');
    [~,Y_grid_idx] = max(Y_grid_prob,[],1);
    Y_grid_original = classLabels(Y_grid_idx);

    gscatter(X_train_ann(:,1),X_train_ann(:,2),Y_train_original)
    hold on

    contour(x1Grid,x2Grid,reshape(double(Y_grid_original),size(x1Grid)), ...
        'LineWidth',1.5)

    xlabel('PCA Component 1')
    ylabel('PCA Component 2')
    title('Decision Regions - PCA + ANN Synthetic Data')

    grid on
    hold off

else

    figure

    gscatter(X_train_ann(:,1),zeros(size(X_train_ann(:,1))),Y_train_original)

    xlabel('PCA Component 1')
    ylabel('Reference Axis')
    title('Training Projection - PCA + ANN Synthetic Data')

    grid on

end

%% =========================================================
% Random individual prediction
%% =========================================================
rng('shuffle')

ind_random = randi(length(Y_test_idx));

x_new = X_test_ann(ind_random,:)';

pred_prob = net(x_new);

[~,pred_idx] = max(pred_prob);

pred_label = classLabels(pred_idx);
real_label = Y_test_original(ind_random);

fprintf('\n--- Random Individual Prediction ANN ---\n');
fprintf('Real class: %d\n', real_label);
fprintf('Predicted class: %d\n', pred_label);

figure

if nPCA == 2
    scatter(X_test_ann(:,1),X_test_ann(:,2),30,Y_test_original,'filled')
    hold on
    scatter(x_new(1),x_new(2),150,'r','filled')

    xlabel('PCA Component 1')
    ylabel('PCA Component 2')
else
    scatter(X_test_ann(:,1),zeros(size(X_test_ann(:,1))),30,Y_test_original,'filled')
    hold on
    scatter(x_new(1),0,150,'r','filled')

    xlabel('PCA Component 1')
    ylabel('Reference Axis')
end

title(['ANN predicts: ',num2str(pred_label), ...
       ' | Real: ',num2str(real_label)])

grid on
hold off
clc
clear all
close all

%% =========================================================
% Cargar datos
%% =========================================================
load('Trainnumbers.mat')

X_original = double(Trainnumbers.image);   % 784 x N
Y = Trainnumbers.label(:);                 % N x 1

[D,N] = size(X_original);

%% =========================================================
% División 70% entrenamiento / 30% prueba
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
% Normalización usando SOLO entrenamiento
%% =========================================================
meant = mean(X_train_raw,2);
stdt = std(X_train_raw,0,2);

stdt(stdt==0) = 1;

X_train_norm = (X_train_raw - meant)./stdt;
X_test_norm  = (X_test_raw  - meant)./stdt;

X_train_norm(isnan(X_train_norm)) = 0;
X_test_norm(isnan(X_test_norm)) = 0;

%% =========================================================
% PCA usando SOLO entrenamiento
%% =========================================================
[transMatcn,Diag] = eig(cov(X_train_norm'));

%% =========================================================
% Selección iterativa del número de componentes PCA
%% =========================================================
componentes = [2 3 5 8 10 12 14 15 17 20 25 30 40 50 60 75 100];

error_train_pca = zeros(length(componentes),1);
error_test_pca  = zeros(length(componentes),1);
error_recon_pca = zeros(length(componentes),1);

Y_train_onehot = full(ind2vec((Y_train + 1)'));
Y_test_onehot  = full(ind2vec((Y_test + 1)'));

for c = 1:length(componentes)

    nPCA_temp = componentes(c);

    transMat_temp = [];

    for i = 1:nPCA_temp
        transMat_temp(i,:) = transMatcn(:,D+1-i)';
    end

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
    net_temp.divideFcn = 'divideind';

    numTrain = size(X_train_ann_temp,1);
    numTest  = size(X_test_ann_temp,1);

    X_all_temp = [X_train_ann_temp; X_test_ann_temp]';
    Y_all_temp = [Y_train_onehot Y_test_onehot];

    net_temp.divideParam.trainInd = 1:numTrain;
    net_temp.divideParam.valInd   = [];
    net_temp.divideParam.testInd  = numTrain+1:numTrain+numTest;

    net_temp.trainParam.epochs = 300;
    net_temp.trainParam.showWindow = false;

    net_temp = train(net_temp,X_all_temp,Y_all_temp);

    Y_train_prob = net_temp(X_train_ann_temp');
    [~,Y_train_pred] = max(Y_train_prob,[],1);
    Y_train_pred = Y_train_pred' - 1;

    Y_test_prob = net_temp(X_test_ann_temp');
    [~,Y_test_pred] = max(Y_test_prob,[],1);
    Y_test_pred = Y_test_pred' - 1;

    error_train_pca(c) = mean(Y_train_pred ~= Y_train);
    error_test_pca(c)  = mean(Y_test_pred ~= Y_test);

    fprintf('nPCA=%d | Train Error=%.4f | Test Error=%.4f | Recon Error=%.4f\n', ...
        nPCA_temp, ...
        error_train_pca(c), ...
        error_test_pca(c), ...
        error_recon_pca(c));

end

%% =========================================================
% Selección automática del mejor número PCA
%% =========================================================
[~,idxBestPCA] = min(error_test_pca);

nPCA = componentes(idxBestPCA);

fprintf('\n--- Mejor número de componentes PCA para ANN ---\n');
fprintf('nPCA seleccionado: %d\n', nPCA);
fprintf('Error prueba mínimo: %.4f\n', error_test_pca(idxBestPCA));
fprintf('Error reconstrucción asociado: %.4f\n', error_recon_pca(idxBestPCA));

%% =========================================================
% Gráfica selección PCA
%% =========================================================
figure

plot(componentes,error_train_pca,'-o','LineWidth',2)
hold on
plot(componentes,error_test_pca,'-s','LineWidth',2)
plot(componentes,error_recon_pca,'-^','LineWidth',2)

xlabel('Número de componentes PCA')
ylabel('Error')
title('Comportamiento de errores PCA + ANN')

legend( ...
    'Error entrenamiento ANN', ...
    'Error prueba ANN', ...
    'Error reconstrucción PCA')

grid on

plot(nPCA,error_test_pca(idxBestPCA),'r*','MarkerSize',12)

hold off

%% =========================================================
% PCA final usando el número seleccionado
%% =========================================================
transMat = [];

for i = 1:nPCA
    transMat(i,:) = transMatcn(:,D+1-i)';
end

ProyTrain = transMat * X_train_norm;
ProyTest  = transMat * X_test_norm;

%% =========================================================
% Reconstrucción final
%% =========================================================
ProyTrainBaseOriginalNorm = transMat' * ProyTrain;
ProyTrainBaseOriginal = ProyTrainBaseOriginalNorm .* stdt + meant;

Emsen = mse(X_train_norm - ProyTrainBaseOriginalNorm);
Emse = mse(X_train_raw - ProyTrainBaseOriginal);

fprintf('\n--- Error de reconstrucción PCA final ANN ---\n');
fprintf('MSE normalizado reconstrucción: %.6f\n', Emsen);
fprintf('MSE reconstrucción original: %.6f\n', Emse);

%% =========================================================
% Visualización imagen original vs reconstruida
%% =========================================================
ind = randi(size(X_train_raw,2));

img_original = lex2img(X_train_raw(:,ind));
img_recon = lex2img(ProyTrainBaseOriginal(:,ind));

figure

subplot(1,2,1)
imshow(img_original)
title('Imagen original')

subplot(1,2,2)
imshow(img_recon)
title(['Reconstrucción con PCA = ', num2str(nPCA)])

%% =========================================================
% Preparar datos para ANN final
%% =========================================================
X_train = ProyTrain';
X_test  = ProyTest';

mu = mean(X_train);
sigma = std(X_train);

sigma(sigma==0) = 1;

X_train_ann = (X_train - mu)./sigma;
X_test_ann  = (X_test  - mu)./sigma;

Y_train_onehot = full(ind2vec((Y_train + 1)'));
Y_test_onehot  = full(ind2vec((Y_test + 1)'));

%% =========================================================
% Entrenamiento ANN iterativo
%% =========================================================
hiddenValues = [10 20 30 40];
epochValues = [100 300];

resultsANN = [];
bestAccuracy = 0;

for i = 1:length(hiddenValues)
    for j = 1:length(epochValues)

        hiddenLayerSize = hiddenValues(i);

        net_temp = patternnet(hiddenLayerSize);

        net_temp.divideFcn = 'divideind';

        numTrain = size(X_train_ann,1);
        numTest  = size(X_test_ann,1);

        X_all = [X_train_ann; X_test_ann]';
        Y_all = [Y_train_onehot Y_test_onehot];

        net_temp.divideParam.trainInd = 1:numTrain;
        net_temp.divideParam.valInd   = [];
        net_temp.divideParam.testInd  = numTrain+1:numTrain+numTest;

        net_temp.trainFcn = 'trainscg';
        net_temp.performFcn = 'crossentropy';

        net_temp.trainParam.epochs = epochValues(j);
        net_temp.trainParam.goal = 1e-6;
        net_temp.trainParam.showWindow = false;

        [net_temp,tr_temp] = train(net_temp,X_all,Y_all);

        Y_pred_train_prob = net_temp(X_train_ann');
        [~,Y_pred_train_class] = max(Y_pred_train_prob,[],1);
        Y_pred_train_class = Y_pred_train_class' - 1;

        trainError = mean(Y_pred_train_class ~= Y_train);

        Y_pred_test_prob = net_temp(X_test_ann');
        [~,Y_pred_test_class] = max(Y_pred_test_prob,[],1);
        Y_pred_test_class = Y_pred_test_class' - 1;

        testError = mean(Y_pred_test_class ~= Y_test);
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
            bestPredANN = Y_pred_test_class;
            bestHiddenLayerSize = hiddenLayerSize;
            bestEpochs = epochValues(j);
        end

    end
end

%% =========================================================
% Tabla resultados ANN
%% =========================================================
resultsANN_table = array2table(resultsANN, ...
    'VariableNames', ...
    {'HiddenNeurons','Epochs','TrainError','TestError','Accuracy'})

fprintf('\n--- Mejor configuración ANN ---\n');
fprintf('Mejor Accuracy ANN encontrado: %.2f%%\n', bestAccuracy);
fprintf('Mejor número de neuronas: %d\n', bestHiddenLayerSize);
fprintf('Mejor número de épocas: %d\n', bestEpochs);

%% =========================================================
% Evaluación final ANN
%% =========================================================
Y_pred_class = bestPredANN;
accuracy = bestAccuracy;
hiddenLayerSize = bestHiddenLayerSize;

fprintf('\n--- Resultados finales ANN con PCA ---\n');
fprintf('Número de componentes PCA seleccionado: %d\n', nPCA);
fprintf('Neuronas capa oculta: %d\n', hiddenLayerSize);
fprintf('Épocas: %d\n', bestEpochs);
fprintf('Accuracy prueba ANN: %.2f%%\n', accuracy);

%% =========================================================
% Matriz de confusión
%% =========================================================
figure

confusionchart(Y_test,Y_pred_class, ...
    'Title','Matriz de Confusión - ANN con PCA', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% =========================================================
% Predicción individual aleatoria ANN
%% =========================================================

rng('shuffle')   % Allows a different random image each run

ind_random = randi(length(Y_test));

x_new = X_test_ann(ind_random,:)';

pred_prob = net(x_new);

[~,pred_label] = max(pred_prob);

pred_label = pred_label - 1;

real_label = Y_test(ind_random);

fprintf('\n--- Predicción individual ANN ---\n');
fprintf('Número real: %d\n', real_label);
fprintf('Número predicho: %d\n', pred_label);

figure

img_lex = X_test_raw(:,ind_random);
img = lex2img(img_lex);

imshow(img)

title(['ANN predice: ',num2str(pred_label), ...
       ' | Real: ',num2str(real_label)])

%% =========================================================
% Gráfica Accuracy - PCA + ANN
%% =========================================================
figure

bar(resultsANN_table.Accuracy)

set(gca,'XTick',1:height(resultsANN_table))

xticklabels(strcat("N=", string(resultsANN_table.HiddenNeurons), ...
                   " / E=", string(resultsANN_table.Epochs)))

xtickangle(45)

ylabel('Accuracy (%)')
xlabel('ANN Configuration')
title('Iterative Results - PCA + ANN')

grid on

[maxAcc, idxBest] = max(resultsANN_table.Accuracy);

hold on
plot(idxBest, maxAcc, 'r*','MarkerSize',12)
hold off

%% =========================================================
% Gráfica Train vs Test Error - PCA + ANN
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
title('Training vs Test Error - PCA + ANN')

legend('Train Error','Test Error')

grid on
hold off
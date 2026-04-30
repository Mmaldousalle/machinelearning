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
componentes = [2 3 5 8 10 12 14 15 20 25 30 40 50 60 75 100];

error_train_pca = zeros(length(componentes),1);
error_test_pca  = zeros(length(componentes),1);
error_recon_pca = zeros(length(componentes),1);

for c = 1:length(componentes)

    nPCA_temp = componentes(c);

    transMat_temp = [];

    for i = 1:nPCA_temp
        transMat_temp(i,:) = transMatcn(:,D+1-i)';
    end

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
% Selección automática del mejor número PCA
%% =========================================================
[~,idxBestPCA] = min(error_test_pca);

nPCA = componentes(idxBestPCA);

fprintf('\n--- Mejor número de componentes PCA ---\n');
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
title('Comportamiento de errores según número de componentes PCA')

legend( ...
    'Error entrenamiento', ...
    'Error prueba', ...
    'Error reconstrucción')

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
% Visualización de imagen original y reconstruida
%% =========================================================
s_img = size(X_train_raw,2);
ind = randi(s_img);

img_lex = X_train_raw(:,ind);
img = lex2img(img_lex);

ProyTrainBaseOriginalNorm = transMat' * ProyTrain;
ProyTrainBaseOriginal = ProyTrainBaseOriginalNorm .* stdt + meant;

img_lex_PPOB = ProyTrainBaseOriginal(:,ind);
img_PPOB = lex2img(img_lex_PPOB);

figure
subplot(1,2,1)
imshow(img)
title('Imagen original')

subplot(1,2,2)
imshow(img_PPOB)
title(['Reconstrucción con PCA = ', num2str(nPCA)])

%% =========================================================
% Error de reconstrucción final
%% =========================================================
Emsen = mse(X_train_norm - ProyTrainBaseOriginalNorm);
Emse = mse(X_train_raw - ProyTrainBaseOriginal);

fprintf('\n--- Error de reconstrucción PCA final ---\n');
fprintf('MSE normalizado reconstrucción: %.6f\n', Emsen);
fprintf('MSE reconstrucción original: %.6f\n', Emse);

%% =========================================================
% Visualización PCA 3D
%% =========================================================
figure
hold on

for i = 0:9
    plot3(ProyTrain(1,Y_train==i), ...
          ProyTrain(2,Y_train==i), ...
          ProyTrain(3,Y_train==i), ...
          '*','Color',rand(1,3))
end

xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('Proyección PCA de datos de entrenamiento')
grid on
hold off

%% =========================================================
% Preparar datos para SVM
%% =========================================================
X_train = ProyTrain';
X_test  = ProyTest';

mu = mean(X_train);
sigma = std(X_train);

sigma(sigma==0) = 1;

X_train_svm = (X_train - mu)./sigma;
X_test_svm  = (X_test  - mu)./sigma;

%% =========================================================
% Entrenamiento SVM iterativo
%% =========================================================
boxValues = [0.1 1 10];
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
% Tabla de resultados SVM
%% =========================================================
results_table = array2table(results, ...
    'VariableNames', ...
    {'BoxConstraint','KernelScale','TrainError','TestError','Accuracy'})

fprintf('\n--- Mejor configuración SVM ---\n');
fprintf('Mejor Accuracy encontrado: %.2f%%\n', bestAccuracy);
fprintf('Mejor BoxConstraint: %.2f\n', bestBox);
fprintf('Mejor KernelScale: %.2f\n', bestKernelScale);

%% =========================================================
% Evaluación del mejor modelo SVM
%% =========================================================

Y_pred = bestPred;

accuracy = bestAccuracy;

fprintf('\n--- Resultados finales SVM con PCA ---\n');
fprintf('Número de componentes PCA seleccionado: %d\n', nPCA);
fprintf('Accuracy prueba: %.2f%%\n', accuracy);

%% =========================================================
% Matriz de confusión del mejor modelo
%% =========================================================

figure

confusionchart(Y_test,Y_pred, ...
    'Title','Matriz de Confusión - Mejor SVM con PCA', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% =========================================================
% Predicción individual aleatoria
%% =========================================================

rng('shuffle')

ind_random = randi(length(Y_test));

x_new = X_test(ind_random,:);
real_label = Y_test(ind_random);

x_new_norm = (x_new - mu)./sigma;

pred_label = predict(svm_model,x_new_norm);

fprintf('\n--- Predicción individual ---\n');
fprintf('Número real: %d\n', real_label);
fprintf('Número predicho: %d\n', pred_label);

figure

img_lex = X_test_raw(:,ind_random);
img = lex2img(img_lex);

imshow(img)

title(['SVM predice: ',num2str(pred_label), ...
       ' | Real: ',num2str(real_label)])
%% =========================================================
% Gráfica Accuracy - PCA + SVM
%% =========================================================
figure

bar(results_table.Accuracy)

set(gca,'XTick',1:height(results_table))

xticklabels(strcat("C=", string(results_table.BoxConstraint), ...
                   " / KS=", string(results_table.KernelScale)))

xtickangle(45)

ylabel('Accuracy (%)')
xlabel('SVM Configuration')
title('Iterative Results - PCA + SVM')

grid on

[maxAcc, idxBest] = max(results_table.Accuracy);

hold on
plot(idxBest, maxAcc, 'r*','MarkerSize',12)
hold off

%% =========================================================
% Gráfica Train vs Test Error - PCA + SVM
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
title('Training vs Test Error - PCA + SVM')

legend('Train Error','Test Error')

grid on
hold off


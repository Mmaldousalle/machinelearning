clc
clear
close all

%% =========================================================
% Comparación final ANN vs SVM - Datos sintéticos
%% =========================================================

% Ajustar estos valores si tus resultados finales son diferentes
accuracy_ann = 98.67;
accuracy_svm = 98.33;

error_ann = 1 - accuracy_ann/100;
error_svm = 1 - accuracy_svm/100;

%% =========================================================
% Gráfica comparativa
%% =========================================================

figure

subplot(1,2,1)

acc_values = [accuracy_ann accuracy_svm];

b1 = bar(acc_values);

set(gca,'XTickLabel',{'ANN','SVM'})

ylabel('Accuracy (%)')
title('Comparación de Accuracy')
ylim([0 100])
grid on

xtips1 = b1.XEndPoints;
ytips1 = b1.YEndPoints;

labels1 = string(acc_values);

offset_acc = 1.5; % separación vertical

text(xtips1,ytips1 + offset_acc,labels1, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontSize',10, ...
    'Color','white')

subplot(1,2,2)

err_values = [error_ann error_svm];

b2 = bar(err_values);

set(gca,'XTickLabel',{'ANN','SVM'})

ylabel('Error')
title('Comparación de Error')
ylim([0 max(err_values)*1.3])
grid on

xtips2 = b2.XEndPoints;
ytips2 = b2.YEndPoints;

labels2 = string(round(err_values,3));

offset_err = max(err_values)*0.05; % offset proporcional

text(xtips2,ytips2 + offset_err,labels2, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontSize',10, ...
    'Color','white')


sgtitle('Comparación Final ANN vs SVM - Datos Sintéticos')
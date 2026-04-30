% Valores reales (ajusta con los tuyos)
accuracy_ann = 93.2;
accuracy_svm = 94.5;

error_ann = 0.071;
error_svm = 0.058;

figure;

% --- Gráfica 1: Accuracy ---
subplot(1,2,1)

acc_values = [accuracy_ann accuracy_svm];

b1 = bar(acc_values);

set(gca,'XTickLabel',{'ANN','SVM'});

ylabel('Accuracy (%)');
title('Comparación de Accuracy');

grid on;

% Tags accuracy
xtips1 = b1.XEndPoints;
ytips1 = b1.YEndPoints;

labels1 = string(acc_values);

text(xtips1, ytips1, labels1,...
    'HorizontalAlignment','center',...
    'VerticalAlignment','bottom');


% --- Gráfica 2: Error ---
subplot(1,2,2)

err_values = [error_ann error_svm];

b2 = bar(err_values);

set(gca,'XTickLabel',{'ANN','SVM'});

ylabel('Error');
title('Comparación de Error');

grid on;

% Tags error
xtips2 = b2.XEndPoints;
ytips2 = b2.YEndPoints;

labels2 = string(err_values);

text(xtips2, ytips2, labels2,...
    'HorizontalAlignment','center',...
    'VerticalAlignment','bottom');
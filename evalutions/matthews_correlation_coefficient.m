function MCC = matthews_correlation_coefficient(y_true, y_pred)
    % Calculate the MCC of a multi-label classifier.（Micro-averaging）
    y_true(y_true<0)=0;
    y_pred(y_pred<0)=0;
    TP = sum(y_true & y_pred, 'all');
    TN = sum(~y_true & ~y_pred, 'all');
    FP = sum(~y_true & y_pred, 'all');
    FN = sum(y_true & ~y_pred, 'all');

    numerator = TP * TN - FP * FN;
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));

    MCC = numerator / denominator;
end
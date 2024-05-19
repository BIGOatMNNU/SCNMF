function BA=Balanced_accuracy(y_true,y_pred)

    % Calculate the balanced accuracy of a multi-label classifier.（Macro-averaging）
%     Parameters:
%         y_true (array-like): The real label array has a shape of (m, n), where m represents the number of samples and n represents the number of labels.
%         y_pred (array-like): The predict label array has a shape of (m, n), where m represents the number of samples and n represents the number of labels.
    y_true(y_true<0)=0;
    y_pred(y_pred<0)=0;
    
    [m, n] = size(y_true);
    ba = 0;
    for i = 1:n
        tp = sum(y_true(:,i) & y_pred(:,i));
        tn = sum(~y_true(:,i) & ~y_pred(:,i));
        fp = sum(~y_true(:,i) & y_pred(:,i));
        fn = sum(y_true(:,i) & ~y_pred(:,i));
        tpr = tp/(tp+fn);
        tnr = tn/(fp+tn);
        if isnan(tpr)
            tpr = 0;
        elseif isnan(tnr)
            tnr = 0;
        end
        ba = ba+(tnr+tpr)/2;

    end
    BA = ba / n;
end
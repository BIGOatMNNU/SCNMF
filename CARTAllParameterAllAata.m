clear;clc;close all;

addpath(genpath('./'));
argoName = "SCNMF";
fileName = "SCNMF_AllParameterAllFeatureRank.mat";
dataFeaturesRank=load(fileName);
fields = fieldnames(dataFeaturesRank);
for i =1:numel(fields)
    field = fields{i};
    value = dataFeaturesRank.(field);
    load(field);
    disp(field);
    if size(value,2)==3
        for ii=1:size(value,1)
            MCC(ii,1) = value{ii,1};
            MCC(ii,2) = value{ii,2};
            BalancedAccuracy(ii,1) = value{ii,1};
            BalancedAccuracy(ii,2) = value{ii,2};
            
            numFeature = length(value{ii,3});
            if numFeature>1000
                numSeleted = round(numFeature * 0.1);
            elseif numFeature<=1000 && numFeature>500
                numSeleted = round(numFeature * 0.2);
            elseif numFeature<=500 && numFeature>100
                numSeleted = round(numFeature * 0.3);
            else
                numSeleted = round(numFeature * 0.4);
            end
            
            selFeature = value{ii,3}(1:numSeleted);
            
            Y = train_target';
            for j=1:size(train_target,1)
                tree = fitctree(train_data(:, selFeature),Y(:,j));
                Y_pred(:,j) = predict(tree, test_data(:, selFeature));
            end
            
            MCC(ii,3)=matthews_correlation_coefficient(test_target', Y_pred);
            BalancedAccuracy(ii,3)=Balanced_accuracy(test_target', Y_pred);
            disp(ii);
        end
        
        filename = 'CARTFresultRecord_'+argoName+'_'+field+'.mat';
        save(filename, 'MCC','BalancedAccuracy');
        clear Y_pred; 
        clear MCC;
        clear BalancedAccuracy;
    elseif len(value)==1
        
    end
end

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
    if size(value,2)==3
        for ii=1:size(value,1)
            HammingLoss(ii,1) = value{ii,1};
            HammingLoss(ii,2) = value{ii,2};
            RankingLoss(ii,1) = value{ii,1};
            RankingLoss(ii,2) = value{ii,2};
            OneError(ii,1) = value{ii,1};
            OneError(ii,2) = value{ii,2};
            Coverage(ii,1) = value{ii,1};
            Coverage(ii,2) = value{ii,2};
            Average_Precision(ii,1) = value{ii,1};
            Average_Precision(ii,2) = value{ii,2};
            
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
            Num=10;Smooth=1;
            [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,selFeature),train_target,Num,Smooth);
            [HammingLoss(ii,3),RankingLoss(ii,3),OneError(ii,3),Coverage(ii,3),Average_Precision(ii,3), Outputs,Pre_Labels]=MLKNN_test(train_data(:,selFeature),train_target,test_data(:,selFeature),test_target,Num,Prior,PriorN,Cond,CondN);
        end
        
        filename = 'MLKNNresultRecord_'+argoName+'_'+field+'.mat';
        save(filename, 'HammingLoss', 'RankingLoss','OneError','Coverage','Average_Precision');
        clear HammingLoss;
        clear RankingLoss;
        clear OneError;
        clear Coverage;
        clear Average_Precision;
        clear MCC;
        clear BalancedAccuracy;
    elseif len(value)==1
        
    end
end

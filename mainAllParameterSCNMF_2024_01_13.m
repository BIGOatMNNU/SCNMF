
clear;clc;close all;

addpath(genpath('./'))
str = {'sample'};
filename = 'SCNMF_AllParameterAllFeatureRank.mat';
for ii = 1:length(str)
    load(str{ii});
    Y = train_target;
    Y(train_target<0) = 0;
    i = 1;
    
    
    for lambda1 = [0.0001,0.001,0.005,0.01,0.05]
        for lambda2 = [0.01,0.1,1,10,100]

            eval([str{ii} '{i,1} = lambda1;']);
            eval([str{ii} '{i,2} = lambda2;']);

            feature_slct = SCNMF(train_data,Y',lambda1,lambda2);
%             feature_slct = thirdPaperMethod2(train_data,Y');

            eval([str{ii} '{i,3} = feature_slct;']);
            i=i+1;
        end
    end
    if exist(filename)~=2
       save(filename, str{ii});
    else
        save(filename, str{ii},'-append');
    end

end

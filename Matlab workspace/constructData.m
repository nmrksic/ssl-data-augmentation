function [augFeatureMatrix, augLabels] = constructData(Train_FM, Train_Labels, Unlabeled_FM)

% This will add the positives from Train_FM and the whole of unlabeled_FM,
% into a new set that we can then apply our SSL methods to!

% Places the unlabeled set first. Then, after any classification, we know
% that (new) postives are the 1s in the first size(Unlabeled_FM,1) rows. 

positives = LabelToArray(Train_Labels);

augFeatureMatrix = sparse(length(positives) + size(Unlabeled_FM, 1), size(Train_FM, 2));

augFeatureMatrix(1 : size(Unlabeled_FM, 1) , :) = Unlabeled_FM(1:size(Unlabeled_FM, 1), :);
augFeatureMatrix(  (size(Unlabeled_FM, 1)+1) : size(augFeatureMatrix, 1) , :) = Train_FM(positives, :);

augLabels = zeros(size(augFeatureMatrix, 1),1);
augLabels((size(Unlabeled_FM, 1)+1) : size(augFeatureMatrix, 1)) = 1;
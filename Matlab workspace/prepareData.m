function [FeatureMatrix, trueLabels] =  prepareData(featureMatrix, positiveSet, mixedSet, allLabels)

% Will return 1...length(positiveSet) in feature matrix as these and label
% them with 1 in trueLabels, regardles of whether they're in the positive
% or unlabeledset - if the actual information is supplied in allLabels -
% this is useful because it allows us to compute the trueLabels we need for
% quality estimation. If we want to hide all positives not in positiveSet,
% we must not label them with 1 in allLabels!

% N.B: positiveSet and mixedSet need not be contiguous! 

% N.B: We are implicitly assuming that the entity set we're expanding is
% always the class labeled 1, and the rest are relabeled to 0!!!

% Positive set contains the array indices of positive elements in the Feature Vector

% FeatureMatrix is the entry * features (binary) matrix 

% MixedSet represents array indices of unlabeled elements

% We are to return new values for featureMatrix and labels etc, to feed
% into I-EM, S-EM, s.t. values not listed are removed

% to pass to I-EM, S-EM, we need to be have featureMatrix, positive and
% mixed set indices - these will be 1:length(positiveSet), and the rest, respectively

% wlog, take all positive values from positiveSet and move them to start of
% the new feature vector. 

% trueLabels will contain information extracted from the actual labels supplied here 
% trueLabels is the subset of allLables for elements in positive and mixed set - 
% however, elements of the positive set are labeled 1, and all the rest are
% labeled 0, to help with subsequent quality estimation.

% Datasets like Reuters will have a feature matrix with elements different
% from 0/1 - we should binarize: 0->0, rest -> 1 TODO TODO TODO TODO TODO TODO TODO TODO TODO


numberOfFeatures = size(featureMatrix, 2);
% the number of columns is the number of features.

numberOfEntries = length(positiveSet) + length(mixedSet);

FeatureMatrix = zeros(numberOfEntries, numberOfFeatures, 'single');
trueLabels = zeros(1, numberOfEntries);

% we will label positives with 1, unlabeled with 0. Take care with NB
% classifier, will change their indexes by +1. 

FeatureMatrix(1:length(positiveSet), :) = featureMatrix( positiveSet, :);
% this will put the features of the positive set in the first
% length(positiveSet) rows of the new feature matrix.

trueLabels(1:length(positiveSet)) = 1;
% The positive sample is definitely positive - left to find positives in MS

FeatureMatrix( (length(positiveSet)+1) : ( length(positiveSet) + length(mixedSet) ), : ) = featureMatrix(mixedSet, :);

trueLabels( (length(positiveSet)+1) : ( length(positiveSet) + length(mixedSet) )) = allLabels(mixedSet);
% what's left is to relabel all those that are not 1 into 0

% we need to binarize the feature matrix - data sets like reuters have
% labels different from 0/1 - so map all non-zeros to 1!
% To do this, we can reuse the binarizePosteriors method:
% This should be done in data pre-processing- too slow to do here!
%for i = 1:size(FeatureMatrix, 1)
%   FeatureMatrix(i, :) = binarizePosteriors(FeatureMatrix(i, :), 0.5);
%   i
%end

for i = 1:length(trueLabels)
    if(trueLabels(i)~=1)
        trueLabels(i)=0;
    end
end
    

function [positiveCentroid, negativeCentroid] = calculateCentroids(FeatureMatrix, PositiveSet, MixedSet, alpha, beta)

% method used with RocSVM to obtain the required centroids

if( nargin < 4) 
    alpha = 16; beta = 4; 
end

numPositives = length(PositiveSet);
numUnlabeled = length(MixedSet);

% Extract the positive and unlabeled parts of the feature matrix
PositiveFeatures(1:numPositives , :) = FeatureMatrix(PositiveSet, :);
UnlabeledFeatures(1:numUnlabeled, :) = FeatureMatrix(MixedSet, :);

% Now, we need to divide each row with its respective norm
% Firstly, calculate the norm of each entry:

positiveNorms = sqrt(sum(PositiveFeatures.^2,2));
unlabeledNorms = sqrt(sum(UnlabeledFeatures.^2,2));


% now, we need to divide each row by the respective scalar
% In order to divide each row by the respective scalar(norm), we need to
% tile the array so that we can divide element by element

PositiveFeatures =  PositiveFeatures ./ repmat(positiveNorms,1,size(PositiveFeatures,2));

% maybe bsxfun can be used to vectorise this! 

%for i=1:size(PositiveFeatures, 1)
 %   PositiveFeatures(i,:) = PositiveFeatures(i,:) / positiveNorms(i);
  %  i
%end

UnlabeledFeatures =  UnlabeledFeatures ./ repmat(unlabeledNorms,1,size(UnlabeledFeatures,2));
%for i=1:size(UnlabeledFeatures, 1)
 %   UnlabeledFeatures(i,:) = UnlabeledFeatures(i,:) / unlabeledNorms(i);
  %  i
%end

% Now, we can calculate the two sum vectors required to calculate the centroids

positiveFactor =  sum(PositiveFeatures) ./ numPositives;
unlabeledFactor = sum(UnlabeledFeatures) ./ numUnlabeled;

% Having these, we are able to calculate the positive and negative centroids

positiveCentroid = single(alpha .* positiveFactor - beta .* unlabeledFactor);
negativeCentroid = single(alpha .* unlabeledFactor - beta .* positiveFactor);
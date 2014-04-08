
function scoreVector = BayesianSet(FeatureMatrix, entitySetIndices, concentrationParameter)

% FeatureMatrix : binary sparse matrix of (entries * features) - hence we
% first transpose it, to obtain features * entries:

FeatureMatrix = FeatureMatrix';

% entitySetIndices : List of entries in the entity set(cluster) that we are trying to expand

% score : the row vector containing the likelihood that the respective entry belongs to the cluster

% concentrationParameter:  Dirichlet concentration parameter (usually set to 2, as is the case in the paper)
if nargin < 3
    concentrationParameter = 2;
end

clusterSize = length(entitySetIndices);
% the number of entries in the training cluster

clusterFeaturesTotal = sum(FeatureMatrix(:,entitySetIndices),2); 
% clusterFeaturesTotal : as defined in the paper, clusterFeaturesTotal is the number of features present in the whole of the sample cluster. We delay applying the logarithm until the next step, to 
% get a more succint/efficient representation. 

% alpha,beta : Beta hyperparameters of prior distribution (column vectors which have values that represent how often specific features appear in entries)

% meanVector: the mean vector of features accross all entries (i.e. the proportion of time that a feature appears in an entry)

numEntries = size(FeatureMatrix,2);
% numEntries is the number of columns (entries)

meanVector = full(sum(FeatureMatrix,2)) / numEntries;

alpha = concentrationParameter * meanVector;

beta  = concentrationParameter * (1 - meanVector);

q = log(1 + clusterFeaturesTotal ./ alpha) - log(1 + (clusterSize - clusterFeaturesTotal) ./ beta);



% q_i represents the row vector q used in the paper, namely, log( (alphanew_j/alpha_j) / (betanew_j/beta_j) )

% in the paper, the coefficient c is defined and added to all the scores. It doesn't change the ranking at all, but isn't costly to compute

c = sum( log(  (alpha+beta)./(alpha+beta+clusterSize)  ) +   log( (beta + clusterSize - clusterFeaturesTotal) ./ beta ) );

scoreVector = c + q' * FeatureMatrix;

% end of file.

 function [finalClass, reliableNegatives, svmClassifierFinal, svmFirstResults, SVMFirstClassifier] = RocSVM(FeatureMatrix, PositiveSet, MixedSet, kkdThreshold, numIter, numClusters, alpha, beta)

% FeatureMatrix is the entry * features (binary) matrix 
% Positive set contains the array indices of positive elements in the Feature Vector
% MixedSet represents array indices of unlabeled elements
% We are implicitly assuming that P and MS cover all rows of FeatureMatrix
% If they are not, prepareData method can take care of that.

% This code implements the Rocchio SVM (with k-clustering) PU learning
% algorithm given in (Learning to Classify Texts using PU data, Li & Liu)

% kkdThreshold is the proportion of points allowed to be on the wrong side
% of the separating hyperplane. The lower it is, the better result the
% algorithm should produce. Furthermore, the lower it is, the longer it
% takes to converge ( exponentially increases the running time, I believe)
% Finally, too low of a value might lead to overfitting. 

% numClusters - parameter used for k-means clustering after Rocchio, default 10

% reliableNegatives - the set of RN identified in step 1 of the algorithm

if(nargin < 6)
     numClusters = 10; alpha = 16; beta = 4;
end

if(nargin < 5)
    numIter = 1000000;
end

if(nargin < 4) 
    kkdThreshold = 0;
end

numEntries = size(FeatureMatrix, 1);

reliableNegatives = Rocchio(FeatureMatrix, PositiveSet, MixedSet, alpha, beta);

if(length(reliableNegatives) == 0) 
    disp('degenerate svm - no RNs')
    finalClass = zeros(1, size(FeatureMatrix, 1));
    return;
end

KMeansCentroids = kmeans(FeatureMatrix(reliableNegatives, :), numClusters, 'start', 'sample', 'emptyaction', 'singleton');

% We need to use this list to make separate lists of entries in each cluster:
% N.B: The indices i in KMeansCentroids are not real, should use
% reliableNegatives(i) instead, so we do that immediately.

temporaryLabels = zeros(numClusters, length(KMeansCentroids));

for i = 1:length(KMeansCentroids)
    temporaryLabels(KMeansCentroids(i), reliableNegatives(i)) = 1;
end

NegativeCluster = cell(numClusters);

for i = 1:numClusters
  NegativeCluster{i} = LabelToArray(temporaryLabels(i, :));
end

% Given the cluster of each entry, we need to produce centroids for each cluster:

positiveCentroids = zeros(numClusters, size(FeatureMatrix, 2), 'single');
negativeCentroids = zeros(numClusters, size(FeatureMatrix, 2), 'single');

for i = 1:numClusters 
   [positiveCentroids(i, :), negativeCentroids(i, :)] = calculateCentroids(FeatureMatrix, PositiveSet, NegativeCluster{i}, alpha, beta);
end

% Subsequently, we iterate through the RN to form the final RN set
% An entry will belong to RN if there exists a negative centroid that it's
% more similar to than all of the positive centroids. Unlike the regular
% Rocchio, here we'll have to do the similarity calculation for all entries
% and all 2*numCluster centroid vectors.

% similarityMatrix[i,j] = cosine_similarity(centroid i, entry j)
similarityMatrixPositive = zeros(numClusters, numEntries, 'single');
similarityMatrixNegative = zeros(numClusters, numEntries, 'single');

% the list of norms of all entries in the FeatureMatrix, needed for calculation
featureNorms = sqrt(sum(FeatureMatrix.^2,2))';

for i = 1:numClusters
    
    % calculate the dot product of the i-th centroid with all entries
    similarityMatrixPositive(i, :) = (sum(FeatureMatrix .* repmat(positiveCentroids(i,:), size(FeatureMatrix, 1), 1), 2));
    similarityMatrixNegative(i, :) = (sum(FeatureMatrix .* repmat(negativeCentroids(i,:) , size(FeatureMatrix, 1), 1), 2));

    % now, divide the dot product by the norm of the respective document:
    similarityMatrixPositive(i, :) = similarityMatrixPositive(i, :) ./  featureNorms;
    similarityMatrixNegative(i, :) = similarityMatrixNegative(i, :) ./ featureNorms;
    
    % in order to complete the cosine similarity calculation, we need to divide by norm of centroid:
    similarityMatrixPositive(i, :)  = similarityMatrixPositive(i, :) / norm(positiveCentroids(i,:));
    similarityMatrixNegative(i, :) = similarityMatrixNegative(i, :) / norm(negativeCentroids(i,:));

end

% Left to determine which elements of RN stay in the RN set.
% For each entry of the unlabeled set, check if the maximal of its
% negative similarities is greater than max of its positive similarities.

% Precompute these maxima/minima:
maxSimilarityPositive(1:numEntries) = max( similarityMatrixPositive(:, 1:numEntries)); 
maxSimilarityNegative(1:numEntries) = max( similarityMatrixNegative(:, 1:numEntries));

similarityDifference = maxSimilarityNegative - maxSimilarityPositive;
% entries with a postive value here belong in RN

% N.B: As in Rocchio, we need to make sure to remove positives from RN:
similarityDifference(PositiveSet) = -1;

% we can use the LabelToArray method - it does >0, whereas the paper
% defines >=0, however as these are continuous scores, should not matter.
reliableNegatives = sort(LabelToArray(similarityDifference));

% NB: TODO: Very high result variability in the number of elements in RN
% Potentially due to k-means clusterin. Examine this. Compare with non k-means.

% Now that Step 1 - RN Set Extraction, is complete, we build the SVM classifiers:

% Now, construct the set Q, which is the set of the remaining unlabeled entries (the set of their indices):

Q_Label = zeros(1, numEntries);

Q_Label(MixedSet) = 1;

Q_Label(reliableNegatives) = 0;

Q = LabelToArray(Q_Label);

% Q will have a much higher percent of positives, (1:2000 ML db yields 60%)
% than the original mixed set did! Q is the "current" Mixed Set!

% Membership of RN and Q are dynamic - we'll be moving from Q to RN

% The RocSVM Method returns either the first classifier produced ( if SVM
% step failed, explained later), or the last classfier. Hence, we will save
% the classifications of the first classifier, and then iterate onwards:

svmLabels = zeros(1, length(PositiveSet) + length(reliableNegatives));
svmLabels(1:length(PositiveSet)) = 1;
svmLabels( ( length(PositiveSet)+1 ) : ( length(PositiveSet)+length(reliableNegatives)) ) = -1;
    
SVMTrainingSet = prepareData(FeatureMatrix, PositiveSet, reliableNegatives, zeros(1, numEntries) );
%svmLabels

SVMFirstClassifier = svmtrain( full(SVMTrainingSet), svmLabels, 'kktviolationlevel', kkdThreshold, 'options', statset('MaxIter', numIter )); %'Display', 'iter' 
SVMClassifier = SVMFirstClassifier;

svmResults = svmclassify(SVMClassifier, FeatureMatrix);
svmFirstResults = binarizePosteriors( svmResults, 1 );

svmResults(PositiveSet) = 0;
svmResults(reliableNegatives) = 0;

W = LabelToArray(-svmResults);

% Now, we iterate; as long as there remain negative elements extracted from Q in the SVM step 
% designate this set W, we remove them from Q and repeat the step.
    
while(not(isempty(W)))
    
    % First, remove W from Q: 
    tempLabels = zeros(1, numEntries);
    tempLabels(Q) = 1;
    tempLabels(W) = 0; % W is a subset of Q, so just remove them from it.
    Q = LabelToArray(tempLabels); % gets the new value of Q

    reliableN = zeros(1, length(reliableNegatives) + length(W)); % the new set of reliable negatives
    reliableN(1:length(reliableNegatives)) = reliableNegatives;
    reliableN( (length(reliableNegatives)+1):(length(reliableN)) ) = W;
    reliableNegatives = reliableN;
 
    %This is where a new iteration technically starts - we've removed W
    %from Q, obtained RN', ready to retradin SVM using P and RN'.

    % svmLabels will contain the training labels for the SVM
    svmLabels = zeros(1, length(PositiveSet) + length(reliableNegatives));
    svmLabels(1:length(PositiveSet)) = 1;
    svmLabels( ( length(PositiveSet)+1 ) : ( length(PositiveSet)+length(reliableNegatives)) ) = -1;

    % SVMTraining set will contain the part of FeatureMatrix with P and RN
    SVMTrainingSet = prepareData(FeatureMatrix, PositiveSet, reliableNegatives, zeros(1, numEntries) );
    
    % Train i-th classifier:
    SVMClassifier = svmtrain( full( SVMTrainingSet) , svmLabels, 'kktviolationlevel', kkdThreshold, 'options', statset('MaxIter', numIter));    
    size(W)
    % Use S_i to classify all the entries:
    svmResults = svmclassify(SVMClassifier, FeatureMatrix);

    % To extract negative elements of Q, we want to first set all entries
    % not in Q (i.e. RN and P) to 0, so they won't be extracted.
    % Need to extract set W: those entries of Q which have negative labels
    % Invert array, because LabelToArray extracts positives' indices.

    svmResultsW = -svmResults;
    svmResultsW(PositiveSet) = 0;
    svmResultsW(reliableNegatives) = 0;
    svmResults = binarizePosteriors(svmResults, 1); % -1 for N needs to be 0
    W = LabelToArray(svmResultsW);
    
end

% Here, what's left is to check whether svmResults( PositiveSet ) has more
% than 5% negatives: if so, the SVM classifier malfunctioned, revert to S1.

svmR = sum(svmResults(PositiveSet)) / length(PositiveSet); % will be the count of positives classified as such by the last classifier

if(svmR < 0.95) 

    finalClass = svmFirstResults; % will have postives labeled 1, negatives -1.
    svmClassifierFinal = SVMFirstClassifier;
    str = ['Reverted to original classifier - SVM iterations converged with ', num2str(svmR)];
    disp(str);
    
else
    
    finalClass = svmResults; % will have postives labeled 1, negatives -1.
    svmClassifierFinal = SVMClassifier;
    str = ['Using the final classifier. SVM iterations converged with ', num2str(svmR)];
    disp(str);
    
end
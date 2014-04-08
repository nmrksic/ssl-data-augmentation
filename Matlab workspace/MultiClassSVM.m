function TestLabels =  MultiClassSVM(TrainFM, TrainLabels, TestFM, kkdThreshold)

% TrainLabels are a single array containing labels 1...N.
% The output is a labelling of all entries in TestFM with 1...N

% To begin with, use a one-vs-all approach, where we train N SVMs, and then
% choose the one with the highest posterior probability as the final label. 


% In principle, one could just use the last approach, but the issue is that
% these different classifiers are trained on different data sets, hence
% representing different problems -> probabilities are not a valid metric
% for comparison, so we only choose to use them as a last resort here.

% As an alternative, we could consider the one-vs-one(i.e. all-vs-all) scheme:
% Build N*(N-1)/2 SVM's that distinguish between pairwise classes, and
% subsequently count the number of votes that each class gets.
% What we'll do is an amalgation of these - if there is more than one label
% claiming this entry ( posterior >= 0.5), we use their intra-classifier to
% decide its membership. This (might) be quite slow, so we will initially
% just choose higest posterior of the one-vs-all classifiers!


% The actual number of different classes:
numClasses = length(unique(TrainLabels));

% The Label arrays that will be used for training binary SVMs:
classLabels = zeros(size(TrainFM, 1), numClasses, 'single');

% This creates these arrays:
for k = 1:numClasses
    classLabels((TrainLabels == k), k) = 1;
end

% Posterior probabilities of testset are determined by the N classifiers:
posteriors = zeros(size(TestFM, 1), numClasses, 'single');

for k = 1:numClasses

    %SVMClassifier = svmtrain(TrainFM, classLabels(:, i), 'kktviolationlevel', kkdThreshold, 'options', statset('MaxIter', 10000000, 'Display', 'final')); 
    %posteriors(:, i) = svmclassify(SVMClassifier, TestFM);
  
end

SVMPairComparison = zeros(numClasses, numClasses, size(TestFM, 1));

for k = 1 : numClasses

    for j = k : numClasses
        
        if k~=j
            
            [reduxFM, reduxLabels] = prepareDataSVM(TrainFM, find(TrainLabels == k), find(TrainLabels == j), TrainLabels); 
            
            SVMClassifier = svmtrain(reduxFM, reduxLabels, 'kktviolationlevel', kkdThreshold, 'options', statset('MaxIter', 10000000)); 
            % now, save all elements' results in cross pair classification
            SVMPairComparison(k, j, :) = svmclassify(SVMClassifier, TestFM);
            SVMPairComparison(j, k, :) = SVMPairComparison(k, j, :); % symmetric matrix!
        end
    end
    
end

TestLabels = zeros(size(TestFM, 1), 1);
    
% accross all entries, we need to count the number of votes they have!

TestLabelsCount = zeros(size(TestFM, 1), numClasses);

for k = 1:size(TestFM, 1)
    % NB THE FIRST CLASS IS DOMINATING EVERYONE, TOTALLY. FIND OUT WHY!
    for i = 1 : numClasses

        for j = 1:numClasses
            
            if( i ~= j) 
                
                TestLabelsCount(k, SVMPairComparison(i, j, k)) =  TestLabelsCount(k, SVMPairComparison(i, j, k)) + 1;
                if(posteriors(k, j) == 1)  
                   TestLabels(k) = j;
                end
                
            end
            
        end
        
    end
end


for k = 1:size(TestFM, 1)

    [~, maX] = max( TestLabelsCount(k, :)); 
    TestLabels(k) = maX;
end


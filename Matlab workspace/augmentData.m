function [newPositivesBayesian, newPositivesSEM, newPositivesRocSVM] = augmentData(FeatureMatrix, Labels, NegativeSet, useBayes, useSEM, useRocSVM, kkt_threshold, numIter)

% this method can be used for augmentation of any class in the dataset:
% all it takes is setting the appropriate Labels array!

if(nargin < 5) 
    useBayes = 1;
    useSEM = 1;
    useRocSVM = 1;
end

if( nargin < 8 )
    kkt_threshold = 0;
    numIter = 1000000;
    disp('Running augmentation with SVM kkt_threshold = 0, numIter = 1000000 - please specify these!')
end

if( nargin < 9)
    numIter = 1000000;
end

disp('starting data augmentation')

% This algorithms returns the three augmented sets of the one given. It
% returns their indices, which can then be used to create a list of labels.
% Subsequently, we can train SVM with this FeatureMatrix and that label
% set, then use it on test data to see how well augmentation did.

% We have the FeatureMatrix and Labels - out of these, we extract positives
% and unlabeled, then apply all three algorithms to them. 

% Then, we identify which of those classified as positives are new ( i.e.
% not present in Labels). Then, put them in the respective set.

% maybe add kddthreshold as parameter to control speed of RocSVM? or some
% parameter to choose methods.

PositiveSet = LabelToArray(Labels);

unlabeled = ones(1, length(Labels));
unlabeled(PositiveSet) = 0;

UnlabeledSet = LabelToArray(unlabeled);

% Having obtained the required sets, deal with sEM:
if(useSEM)
    Results = SpyEM(FeatureMatrix, PositiveSet, UnlabeledSet);
    Results = binarizePosteriors(Results);
    Results(PositiveSet) = 0; % Remove old positives, to get out just new ones.
    newPositivesSEM = LabelToArray(Results);
    disp('augmented sem')
else
    newPositivesSEM = [];
end

if(useBayes)
    
    numToExtract = fix ( (length(PositiveSet) * size(FeatureMatrix, 1)) / (length(PositiveSet) + length(NegativeSet))) - length(PositiveSet);
    % limit the potential amount of noise added!

    % useIterative:
    %{ 
    scores = iterativeBayesianSets(FeatureMatrix, PositiveSet, 2);
    disp('xxx')
    
    
    newPositivesBayesian = scores(1:min(length(scores), numToExtract)); 
    %}
    
    % use singleBayes:  
    scores = BayesianSet(FeatureMatrix, PositiveSet, 2);
    [~, idx] = sort(scores, 'descend');
    newPositivesBayesian = idx(1:min(length(scores), numToExtract));
    newPositivesBayesian = setdiff(newPositivesBayesian, PositiveSet);
    
    %                   
    
    
    disp('augmented bayes')
    strr = ['Size of new Positives via Spy-EM is:', num2str(length(newPositivesSEM)), '\n and the number of new positives via Bayesian sets is:', length(newPositivesBayesian)];
    disp(strr)
else
    newPositivesBayesian = [];
end

% Deal with RocSVM:
if(useRocSVM)

    lenPos = sum(Labels);
    
    Results =  RocSVM(FeatureMatrix, PositiveSet, UnlabeledSet, kkt_threshold, numIter);
    Results = binarizePosteriors(Results); % using threshold = 0.5
    
    lenPos2 = sum(Results);
    
    Results(PositiveSet) = 0; % Remove old positives, to get out just new ones.
    newPositivesRocSVM = LabelToArray(Results);
   % disp('augmented rocsvm')

else
    newPositivesRocSVM = [];
end
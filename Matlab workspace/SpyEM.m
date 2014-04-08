function [finalClassPosteriors, P, U, N, iterationCount] = SpyEM(FeatureMatrix, PositiveSet, MixedSet, negativeThreshold, realLabels)

% if the realLabels are not passed as argument, then we won't be printing
% out precision, recall and fscore at each step of the algorithm !!!

if( nargin < 4)
    negativeThreshold = 0.15;
% entries with posterior Probability less than negativeThreshold are put in 
% the reliable negative set - the paper uses 0.15 by default!
end


% We want to return the labels of the unlabeled cases, once the iterations
% of creating the Naive Bayesian classifiers have converged. Or the
% posterior probabilities (maybe binarized) of the final classifier

% Positive set contains the array indices of positive elements in the Feature Vector

% FeatureMatrix is the entry * features (binary) matrix 

% MixedSet represents array indices of unlabeled elements

% We are implicitly assuming that P and MS cover all rows of FeatureMatrix

% If they are not, prepareData method can take care of that.

numPositives = length(PositiveSet);

permutationOfPositives = randperm(numPositives);

setPositives(1:numPositives) = PositiveSet(permutationOfPositives);

% in the paper, s = 10% is used for the spy set.

reducedPositives = setPositives(1:round(numPositives*9/10));

spyPositives = setPositives( (round(numPositives*9/10)+1):numPositives);

mixedSet = zeros(1, length(spyPositives) + length(MixedSet) );

mixedSet(1:length(MixedSet)) = MixedSet;

mixedSet(  (length(MixedSet)+1):length(mixedSet) ) = spyPositives;

% mixedSet contains all of the mixed set + 10% of positives at the end

numberOfSpies = length(spyPositives);
% this is the number of elements included as spies at the end of (new) mixedSet

reorderedFeatureMatrix(1:length(reducedPositives), :) = FeatureMatrix(reducedPositives, :);
reorderedFeatureMatrix( length(reducedPositives)+1 : (length(reducedPositives) + length(mixedSet)), :) = FeatureMatrix(mixedSet, :);

initialEMLabels = InitialEM (reorderedFeatureMatrix, 1:length(reducedPositives), length(reducedPositives)+1: (length(reducedPositives) + length(mixedSet)));
% we're accessing indexes at each point, reducedPositives and mixedSet
% don't have to be in order in the featureMatrix to work - indices are fine

%NB: initialEMLabels will contain the posterior probabilities of each row
%of the Feature Matrix - it won't be indexed in the order in which we
%constructed mixedSet or what ever.

% IEMLabelsIndexed = zeros(length(initialEMLabels));

% IEMLabelsIndexed(1:length(reducedPositives)) = initialEMLabels(reducedPositives);
% IEMLabelsIndexed(  (length(reducedPositives)+1) : (length(initialEMLabels) - numberOfSpies) ) = initialEMLabels(mixedSet);


% [1,length(reducedPositives)] is the set of posteriors of positives used in training the
% classifier, and the rest is the mixedSet. Indexing is the same as in
% reducedPositives, so IEMLabelsIndexed(1) is reducedPositives(1)...
% and IEMLabelsIndexed(length(reducedPositives)+1) is mixedSet(1) 

% Now, what's left is to put all members of initialEMClasses with posterior
% probability <= t, into the Reliable Negative set, and restore the
% original positive set - we can just use the old one, PositiveSet

initialEMLabels = initialEMLabels( (1+length(reducedPositives)) : (length(initialEMLabels) - numberOfSpies));
% ReducedPositives and Spypositives are at the beggining and at the end of 
% initialEMLabels, and there are numberOfSpies of them!! Therefore, we just 
% need to "truncate" this array to obtain the middle - which is mixedSet!

[valuesPosterior, valuesIndices] = sort(initialEMLabels);
% we sort by posterior probabilities so that we can easily identify the
% likely negatives set! No need for binary search, linear is fast enough!

bound = length(valuesPosterior);

for i=1:length(valuesPosterior)
    if(valuesPosterior(i)>=negativeThreshold)
        bound = i - 1;
        break;
    end
end

% the bound will represent the point at which we go from N to U

% All values with the valuesPosterior(i) <= thresholdNegative should go to
% negative set, however the index that should be going to the negative set
% needs to be obtained from its valuesIndices(i). if it's <=
% length(ReducedPositives) OR > length(reducedPositives) + length(mixedSet)
% that means that it's originally from the positive set - it should be
% mapped to one of these indices - but we don't have to mess with these, 
% as they won't be going into either N or U, they'll become P, and can be
% copied from the original.
% What we need to do is, for those that do not meet the above condition,
% i.e. those that are originally from the mixedSet, we need to move them
% either to negative or unlabeled, BUT WITH THEIR ORIGINAL INDEX!!!
% To achieve all this, we need just reorder the feature matrix before
% feeding into IEM, much, much neater!!!

% since valuesIndices are indexes in the mixedSet, which is an array of
% indices, that means that we need to feed these into N/U with these indices
negativeSet = MixedSet(sort(valuesIndices(1:bound)));
unlabeledSet = MixedSet(sort(valuesIndices((bound+1):(length(valuesIndices)))));


% the third set we need is the Positive Set - we can use the original one!

positiveSet = PositiveSet;
% just to achieve naming uniformity

% sanity check to see if all elements were divided into these three sets,
% by checking that the sum is what it should be.
% d = length(positiveSet) + length(negativeSet) + length(unlabeledSet);
% isequal( sum(positiveSet) + sum(negativeSet) + sum(unlabeledSet), d*(d+1)/2)
 
P = positiveSet;
U = unlabeledSet;
N = negativeSet;

% what the sEM algorithm does next is train an NB classifier only using the
% sets P and N. This classifier is then used to classify all of U. Then,
% given these posteriors -> labels, we iteratively construct new NB
% classifiers, as in iEM, using all three sets with their labels, until
% they converge. We can't reuse the iEM code here, as it resets the positive 
% set to 1 in each iteration, so elements of U labeled 1 by the initial NB
% won't be able to change label - which is incorrect!!!

%TODO CHECK IF THIS PN CLASSIFIER HAS BEEN USED IN THE RIGHT WAY

newLabels = zeros(1, length(positiveSet)+length(negativeSet)+length(unlabeledSet));
newLabels(1:length(positiveSet)) = 1;
% newLabels are the labels prepared for the new NB classifier, at first
% longer to survive prepareData which needs it to be of same lenght as FeatureMatrix

clear reorderedFeatureMatrix;

[reorderedFeatureMatrix, newLabels] = prepareData(FeatureMatrix, positiveSet, negativeSet, newLabels);
% obtain the feature matrix just for P and N, get newLabel to resize to get
% rid of its elements from unusedSet.

PN_NBClassifier = NaiveBayes.fit(reorderedFeatureMatrix, newLabels, 'Distribution', 'mn');
%train the classifier using only P and N

posteriors = PN_NBClassifier.posterior(reorderedFeatureMatrix);
% Now, we need to classify all elements of U, N according to the new
% classifier, and then run the EM algorithm, fixing posteriors of P to 1.


labelsEM = zeros(1, length(positiveSet)+length(negativeSet)+length(unlabeledSet));
% this will be the list of labels for our EM algorithm

labelsEM(positiveSet) = 1;
% the elements in the positiveSet are fixed to have posterior probability 1

if(~isempty(negativeSet) )% if the negative set is NOT empty:
    labelsEM(negativeSet) = binarizePosteriors(posteriors( length(positiveSet)+1:length(positiveSet)+length(negativeSet) ,2));
    disp('negatives not empty, all good')
end

% the negative ones are free to be relabeled, as will be those from U
posteriorsU = PN_NBClassifier.posterior(FeatureMatrix(unlabeledSet, :));

if(size(posteriorsU, 2) < 2) % in case empty or fully determined
    finalClassPosteriors = labelsEM; % if determined, U is all 0
    disp('massive bug2')
    return;
end
   
labelsEM(unlabeledSet) = binarizePosteriors ( posteriorsU(:,2));

% Now, all that's left is to run EM until it converges: retrain NB given
% these labels, obtain posteriors, binarize, reset P to 1, repeat.

oldLabels = zeros(size(labelsEM));

numIterations = 0;
% TODO: have this as a return parameter.

while ( not(isequal(oldLabels, labelsEM)))

    if(numIterations > 30) % this is probably close to converging within 2,3 iterations, let alone 20, no need to continue
        str = ['Cut SEM optimization after 30 iterations - the number of different labels is:', num2str(sum(abs(oldLabels - labelsEM)))];
        disp(str)
        break;
    end
    
    EM_NBClassifier = NaiveBayes.fit(FeatureMatrix, labelsEM, 'Distribution', 'mn');
    
    posteriorProbabilities = EM_NBClassifier.posterior(FeatureMatrix);
    
    oldLabels = labelsEM;
    
    labelsEM = binarizePosteriors(posteriorProbabilities(:,2));
    
    labelsEM(positiveSet) = 1;
    % make sure to keep these fixed to 1.
        
    numIterations = numIterations + 1;
    
    % The next two lines, and the realLabels that might be passed in as
    % argument, can be used to track how subsequent classifiers improve the
    % estimation metrics! will only write this if we pass these as argument
    
end

% seems to work and produce the same results as iEM, at least for this exp

iterationCount = numIterations;
finalClassPosteriors = posteriorProbabilities(:,2);

% the question remains - do we fix the posteriors of positives to be 1?
% the paper specifies that in this procedure, we are always to keep the
% posteriors of these to 1 - hence the answer is yes!

finalClassPosteriors(positiveSet) = 1;
% this ensures that, when these are subsequently binarized, we get that the
% positives always have class 1 - as is required by conditions in the paper


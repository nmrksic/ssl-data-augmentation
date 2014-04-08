function [finalClass, iterations] = InitialEM(FeatureMatrix, PositiveSet, MixedSet)

% NB: We are always handling indexes of PositiveSet, MixedSet - no need for
% these to be contiguous, but they do cover all of the feature matrix rows
% between them...

% We want to return the posterior probabiliteies of labels of the unlabeled cases, 
% once the iterations of creating the Naive Bayesian classifiers have converged.

% Positive set contains the array indices of positive elements in the Feature Vector

% FeatureMatrix is the entry * features matrix 

% MixedSet represents array indices of unlabeled elements

% We are implicitly assuming that P and MS cover all rows of FeatureMatrix
% This is done pre-I-EM with prepareData

% This method returns the posteriorProbabilities of the final classifier.
% It shouldn't return newLabels, as that (amongst other things) skews the
% recall of the sample set to 1, which isn't realistic, as we don't
% actually have that classifier built, but were biasing towards the
% positive set in every iteration, as explained in the paper.

% First, we build the initial Naive Bayesian classifier, by labeling
% Positive elements as 1, and the negative ones as 0:

labelCount = (length(PositiveSet)+length(MixedSet));
% the total number of labels

initialLabels = zeros(1, labelCount);

for i = 1:length(PositiveSet)
    initialLabels(PositiveSet(i)) = 1;
end

% this array of initial labels will be used subsequently to override NB
% classifiers when assigning labels to members of positive set
  
InitialNBClassifier = NaiveBayes.fit(FeatureMatrix, initialLabels, 'Distribution', 'mn');

posteriorProbabilities = InitialNBClassifier.posterior(FeatureMatrix);

oldLabels = initialLabels;

newLabels = binarizePosteriors(posteriorProbabilities(:, 2));

newLabels(PositiveSet) = 1;

% Repeated iterations of creating new classifiers until their label assignements converge. 
% They will have converged when oldLabels == newLabels - as these are used to generate posteriors, 
% that means that the posterior probabilities will have converged as well.

iterationCount = 1;
% we might be interested at some stage to see how long it takes for this step to converge

NBClassifier = InitialNBClassifier;

while( not(isequal(oldLabels, newLabels)) ) 
    
    % mean(posteriorProbabilities(:,1))
    % mean(posteriorProbabilities(:,2))
    
    if(iterationCount>30)  % this is probably close to converging within 2,3 iterations, let alone 20, no need to continue
            diff  = sum(abs(oldLabels - newLabels)) 
            break;
    end
    
    NBClassifier = NaiveBayes.fit(FeatureMatrix, newLabels, 'Distribution', 'mn');
    
    % create the New NB classifier according to the new labels
    posteriorProbabilities = NBClassifier.posterior(FeatureMatrix);
    % obtain its new posterior probabilities...

    % NB: 1 is label for P, 0 is label for N. However, classifier will have
    % the probs for 1 in 2, and probs for 0 in 1. Source of confusion!!
        
    oldLabels = newLabels;
    % remember the labels used for creating this classifier
   
    % NB: Reset newLabels to 0s, changed to vector operations here, works
    
    newLabels = binarizePosteriors(posteriorProbabilities(:,2));
    
    newLabels(PositiveSet) = 1;
    
    % ...and then use posterior probabilities to obtain new labels.
    iterationCount = iterationCount + 1;
    
end
 
iterations = iterationCount;

finalClass = NBClassifier.posterior(FeatureMatrix);

finalClass = finalClass(:,2);


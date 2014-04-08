function [precision, recall, fscore, precisionM, recallM, fscoreM] = qualityOfClassifier(newLabels, trueLabels)


% we let the confusionMat deal with label ordering - we are interested in
% the second one, as we label P with 1, U with 0 - so return second class,
% and then, if need be, the whole matrix.
% we will just return precision, recall, fscore of each class - we'll be
% interested in the parameters of the second one, usually

% function to calculate precision, recall, fscore of classification

% in the context of PU learning, we are conserned specifically with
% prediction and recall of the target cluster, so we output 
% the prediction and recall of every class as well.

% The precision/recall in identifying the positive elements is
% substantially different than overall parameters of classifier, and this is 
% most relevant observation for our research. Hence, return all classwise
% parameters.

cMat = confusionmat(trueLabels, newLabels);
% i,j means is actually i but is classified as j
numberOfClasses = length(unique(trueLabels));

recallM = zeros(1, numberOfClasses);
precisionM = zeros(1, numberOfClasses);

for i = 1:numberOfClasses
    recallM(i) = cMat(i,i) / sum(cMat(i,:));
end % recall is the proportion retrieved

for i = 1:numberOfClasses
    precisionM(i) = cMat(i,i) / sum(cMat(:,i));
end

recall = recallM(2);

precision = precisionM(2);

fscoreM = 2 * (precisionM .* recallM) ./ (precisionM + recallM);

fscore = fscoreM(2);

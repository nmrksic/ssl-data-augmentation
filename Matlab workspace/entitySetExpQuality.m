function [precision, recall, fscore] = entitySetExpQuality(newLabels, trueLabels)

% function to extract the relevant prec, recall, fscore from qualOfClass

% usually, we label U with 0, P with 1, resulting in cmat using the 2nd
% column for classifications of the target cluster. Tested this.

[~,~,~,pr,rc,fsc] = qualityOfClassifier(newLabels, trueLabels);

recall = rc(2);

precision = pr(2);

fscore = fsc(2);


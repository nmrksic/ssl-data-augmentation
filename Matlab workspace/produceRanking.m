function produceRanking(scoreVector, fileName, numberOfResults) 

% scoreVector is the list of likelihoods produced by the Bayesian Sets algo

% fileName: pathname to the file which contains the list of entry names

% resultNumber: the number of results to be displayed

if nargin < 3
  numberOfResults = 10;
end

entryNames = importdata(fileName);

[~, sortedIndices] = sort(scoreVector, 'descend');

% sortedIndices will contain where in the original list the i-th most likely
% candidate for membership of the cluster is.

for idx = 1:numberOfResults
    current = sortedIndices(idx);
    fprintf('Score: %.4g; Pos: %d;  Name: %s \n', scoreVector(current), current, entryNames{current});
end
    
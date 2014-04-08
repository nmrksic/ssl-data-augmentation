function [sortedIndices, sortedScores] = produceRankingDirect(scoreVector, entryNames, numberOfResultsPrint) 

% scoreVector is the list of likelihoods produced by the Bayesian Sets algo

% entryNames: This is the alternative produceRanking method, when the class
% labels are not given in a separate file, but are loaded in the Matlab
% workspace instead.

% resultNumber: the number of results to be displayed

if nargin < 3
    numberOfResultsPrint = 0;
end


[sortedScores, sortedIndices] = sort(scoreVector, 'descend');

% sortedIndices will contain where in the original list the i-th most likely
% candidate for membership of the cluster is.

for idx = 1:numberOfResultsPrint
    current = sortedIndices(idx);
    fprintf('Score: %.4g; Pos: %d;  Class: %s \n', scoreVector(current), current, entryNames(current));
end
    
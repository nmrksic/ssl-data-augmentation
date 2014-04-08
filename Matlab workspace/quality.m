function [values] = quality(NewLabels, TrueLabels)

if(size(NewLabels, 1) ~= size(TrueLabels, 1))
    NewLabels = NewLabels';
end

truePositives = sum( (inverseMatrix(NewLabels - TrueLabels)) .* NewLabels  );

falsePositives = sum(NewLabels) - truePositives; % those perceived as P but actually N/U.

falseNegatives = sum(TrueLabels) - truePositives; % postives actually perceived as negatives

trueNegatives = length(NewLabels) - truePositives - falseNegatives - falsePositives;

% total positives = truePos + falseNeg
% precision = truePos / (truePos + falsePos)  -  specificity
% recall = truePos / (truePos + falseNeg) - also called sensitivity

precision = truePositives / ( truePositives + falsePositives );
recall = truePositives / ( truePositives + falseNegatives );
fscore = 2 * precision * recall  / (precision + recall);



confusionmatt = confusionmat(TrueLabels, NewLabels)

accurraacy = (trace(confusionmatt) ) / sum(sum(confusionmatt))

values = zeros(1,3); 

values(1,1) = precision; 
values(1,2) = recall; 
values(1,3) = accurraacy;

%values(1,4:5) = confusionmatt(1,1:2);
%values(1,6:7) = confusionmatt(2, 1:2); 

% we need to handle invalid values

if(truePositives + falsePositives == 0)
    values(1, :) = 2; % meaning none are classifier as positive, even wrongly! used to be NaN 0 NaN
end

if(precision==0 && recall==0)
    values(1, :) = 3; % means all actual positives are are classified as zeros??
end

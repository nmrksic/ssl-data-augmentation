function [reduxFM, reduxLabels, reduxUnLabels, reduxUnLabelsInvert] = createRedux(FeatureMatrix, Labels, sample, bias)

% NegativeSet is the listing of negatives in reduxFM
disp('starting redux creation with sample= ')
sample
% First, create the Positive and Negative set, that we will be sampling from:

PositiveSet = LabelToArray(Labels);

negatives = ones(1, length(Labels));
negatives(PositiveSet) = 0;

NegativeSet = LabelToArray(negatives);

numPositives = length(PositiveSet);
numNegatives = length(NegativeSet);


% based on sample and bias, we can determine what proportion of postives and negatives we need to sample:

% we need to renormalize, so that ( sample*bias * numPositives + sampleN * numNegatives)/normalization = sample * (numPositives + numNegatives):
normalization =  sample * (numPositives + numNegatives) / (sample * bias * numPositives + sample * numNegatives);

sampleP = fix((sample * bias * numPositives) / normalization);
sampleN = fix((sample*numNegatives) / normalization);

% if any of them are empty - we need to capture this as an error
if(sampleP==0 || sampleN==0) 
    sampleP 
    sampleN
    return;
end

% Now, we sample sampleP from the PositiveSet, and sampleN from the Negative set:

PositiveSamples = randsample(PositiveSet, sampleP);
NegativeSamples = randsample(NegativeSet, sampleN);

% The remaining elements (those not sampled), should be in the new "unlabeled" set:

UnlabeledPositives = setdiff(PositiveSet, PositiveSamples); % those from PositiveSet not in PositiveSamples
UnlabeledNegatives = setdiff(NegativeSet, NegativeSamples); % those from NegativeSet not in NegativeSamples

% sanity check(both should be 0):
if ( length(UnlabeledNegatives) + length(NegativeSamples) - length(NegativeSet) ~= 0)
    return; % TODO: add warning message
end

if ( length(PositiveSamples) + length(UnlabeledPositives) - length(PositiveSet) ~= 0)
    return; % TODO: add warning message
end


% Now, we want to create the feature matrix and the labels of this set:

reduxFM = zeros(sampleP+sampleN, size(FeatureMatrix, 2), 'single');
reduxFM(1:sampleP, :) = FeatureMatrix(PositiveSamples, :);
reduxFM( (sampleP+1) : (sampleP + sampleN), :) = FeatureMatrix(NegativeSamples, :);

reduxLabels = zeros(1, sampleP + sampleN);
reduxLabels(1:sampleP) = 1;

% There is no need to create a new matrix for the unlabeled set - we 
% actually have to use the full FM, we just need a new set of labels.

reduxUnLabels = zeros(1, length(Labels));
reduxUnLabels(PositiveSamples) = 1;

reduxUnLabels = reduxUnLabels';

reduxUnLabelsInvert = zeros(length(Labels), 1);
reduxUnLabelsInvert(NegativeSamples) = 1;
% leave all of the unlabeled set as unlabeled!!!

disp('done creating redux')
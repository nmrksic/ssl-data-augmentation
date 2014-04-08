function binarizedPosteriors = binarizePosteriors(posteriors, threshold)

% this function takes an array od continuous values in [0,1] and returns
% 1 in each position where probability >= 0.5, otherwise 0

% threshold is used if we want to binarize with some biased probability, as
% in when picking out the negativeSet in sEM

if nargin < 2
  threshold = 0.5;
end

binarizedPosteriors = zeros(1, length(posteriors));

for i = 1:length(posteriors)
    if(posteriors(i)>=threshold) 
        binarizedPosteriors(i) = 1;
    end
end


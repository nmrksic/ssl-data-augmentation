function positiveIndices = LabelToArray(Labels)

% this function takes an binary array, and returns the list of indices > 0
% in a binary array, this will mean all 1s (RocSVM post k-means cluster extraction)
% in a continuous values array, this will mean >0, RocSVM maximisation
% in principle, can be used to extract the indices of the part of the array
% which has values > 0. 

[values, indices] = sort(Labels);

% idx will contain the location of the first 1
idx = -1;
for i = 1:length(values)
    if values(i)>0
        idx = i; break;
    end
end

if(idx~=-1)
    positiveIndices = indices(idx:length(indices));
else
    positiveIndices = int32.empty;
end

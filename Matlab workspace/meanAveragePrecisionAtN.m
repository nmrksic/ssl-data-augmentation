function precision = meanAveragePrecisionAtN (rankedIndices, trueLabels, N)

sumPrecisions = 0;

for i = 1:N
   sumPrecisions = sumPrecisions + precisionAtN(rankedIndices, trueLabels, N);
end

precision = sumPrecisions / N;
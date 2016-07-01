%This is supplementary code for SIGGRAPH submission #248
%Color Compatibility for Large Datasets


%set the code directory...
codeRoot= [pwd, '/'];
addpath([codeRoot])
addpath([codeRoot,'data/'])
addpath([codeRoot,'circstat/'])
addpath([codeRoot,'glmnet_matlab/'])
load('learned_model.mat', 'fit', 'offsets', 'scales');

%create features
% [datapoints data] = createDatapoints(dataset,maxNumberOfDatapoints);
% input_colors = data(testingPts, :, :)
% input_colors = [];
%input_colors = [
    %0.3451    0.5490    0.5490;
    %0.2863    0.4196    0.4510;
    %0.7490    0.8196    0.8510;
    %0.5765    0.6824    0.7490;
    %0.4510    0.2118    0.2549;
%];
input_colors = transpose(input_colors);
input = [];
for i=1:3
    input(:, :, i) = input_colors(i, :);
end
% input_colors(:, :, 1) = [0.3451    0.2863    0.7490    0.5765    0.4510];
% input_colors(:, :, 2) = [0.5490    0.4196    0.8196    0.6824    0.2118];
% input_colors(:, :, 3) = [0.5490    0.4510    0.8510    0.7490    0.2549];
[allFeatures]= createFeaturesFromData(input,30000);
names = fieldnames(allFeatures);
% Create empty feature vector with the right size
features=[];
for i=1:size(names,1)
   features=[features allFeatures.(cell2mat(names(i)))];
end
% Whiten features
for i=1:size(features,2)
    features(:,i) = features(:,i)-offsets(i);
    features(:,i) = features(:,i)./scales(i);
end

testingPrediction = glmnetPredict(fit, 'response', features);

fileID = fopen('result.txt', 'w');
fprintf(fileID, '{"result": %f}', testingPrediction);
fclose(fileID);

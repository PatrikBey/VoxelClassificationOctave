% presentation script:
% 'Mathematical Software: Statistical learning in octave'
%
% author: Patrik, Bey; beypatri@gmail.com
%
% date: 2018.06.12

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize workspace						   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set working directory

cd ~/Studies/MSc_Thesis/Data/Test_Data/subj_1101

% add path to .mex files
addpath '~/libsvm/matlab'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions									   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [pred,SVMmodel] = performClass (d,c,k)
	%define size of data input
	dim = size(d);
	% create label and index vectors for later classification
	labels = [repmat(-1,dim(1)/2,1);repmat(1,dim(1)/2,1)];
	id = 1:length(labels);
	% define class size / assuming balanced classes!
	classsize = dim(1)/2;
	%LOOCV for classificationvia SVM
	for i = 1:dim(1)
	 if (i < classsize)
	  ex = [i,classsize+floor(rand(1)*classsize)];
	 else
	  ex = [floor(rand(1)*classsize),i];
	 endif
	idtrain = randperm(length(id));
	idtrain = idtrain(idtrain ~= ex(1));
	idtrain = idtrain(idtrain ~= ex(2));
	train = double(d(idtrain,:));
	labtrain = double(labels(idtrain));
	test = double(d(i,:));
	SVMmodel = svmtrain(labtrain,train,['-c ' mat2str(c) ' -t ' mat2str(k)]);
	pred(i) = svmpredict(0,test,SVMmodel);
	end
endfunction

function [acc,sen,spec,ppv,npv] = compPerformance(pred,lab)
	% compute variables
	dim = length(pred);
	% create confusion matrix for computation
	% initialize values
	tn = 0;
	tp = 0;
	fn = 0;
	fp = 0;
	% compute true / false predicitons
	for i = 1:dim
	 if (i <(dim/2+1) )
	  if (pred(i) == lab(i))
	   tn = tn+1;
	  else
	   fp =fp+1;
	  endif
	 else
	  if (pred(i) == lab(i))
	   tp = tp+1;
	  else
	   fn = fn+1;
	  endif
	 endif
	end
	% classification preformance metrics
	% overall performance accuracy
	acc = (tp+tn) / dim(1);
	% true positive rate SENSITIVITY
	sen = tp/(dim(1)/2);
	% true negative rate SPECIFICITY
	spec = tn/(dim(1)/2);
	%PRECISION:positive prediction value
	ppv = tp/(tp+tn);
	%negative prediction value
	npv = tn/(tp+tn);
endfunction



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data loading / subsetting					   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% load data
dat = load('REST_ts.mat');

% select relevant brain regions
% amygdala for emotion processing
% inferior parietal lobe for rational mathematical logic
amyL = dat.REST_ts.norm.ROI_41;
	amyL = amyL(5:204,:);
amyR = dat.REST_ts.norm.ROI_42;
	amyR = amyR(5:204,:);
iplL = dat.REST_ts.norm.ROI_61;
	iplL = iplL(1001:1200,:);
iplR = dat.REST_ts.norm.ROI_62;
	iplR = iplR(1001:1200,:);

% combine to on data matrix
data = [iplR;iplL;amyR;amyL];
dim = size(data);
labels = [repmat(-1,dim(1)/2,1);repmat(1,dim(1)/2,1)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classification 							   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define cost values
cost = [0.0001,0.1,1,1000];
kernel = [0,2,3];

gridACC = zeros(length(cost),length(kernel));

tic
for k = kernel
 for c = cost
  [pred,model] = performClass(data,c,k);
  gridACC(find(cost == c),find(kernel == k)) = compPerformance(pred,labels);
 end
end
time = toc/60;
% time needed for run on my PC (UB 18.04; i5-7200 4*3.1GHz;8GB RAM)
% ~59 minuten

% gridACC =
%   0.99625   0.50375   0.49875
%   0.99625   0.50375   0.49875
%   0.99625   0.50750   0.49625
%   0.99625   0.51125   0.49875



% plotting of results
% for each kernel function same cost parameters(c=0.1 and c=1000)
k = 0
[pred,model] = performClass(data,c,k);

s = sign(model.sv_coef);
C = [((s+2)/3), ones(length(s),1), ones(length(s),1)]; 
C = hsv2rgb(C);  

scatter(model.sv_indices,model.sv_coef,50,C,'filled','MarkerEdgeColor','k');
title('Support Vector weights and signs for linear kernel')

k = 2
[pred,model] = performClass(data,c,k);

s = sign(model.sv_coef);
C = [((s+2)/3), ones(length(s),1), ones(length(s),1)]; 
C = hsv2rgb(C);  

scatter(model.sv_indices,model.sv_coef,50,C,'filled','MarkerEdgeColor','k');
title('Support Vector weights and signs for rbf kernel')

k = 3
[pred,model] = performClass(data,c,k);

s = sign(model.sv_coef);
C = [((s+2)/3), ones(length(s),1), ones(length(s),1)]; 
C = hsv2rgb(C);  

scatter(model.sv_indices,model.sv_coef,50,C,'filled','MarkerEdgeColor','k');
title('Support Vector weights and signs for sigmoid kernel')
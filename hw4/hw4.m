
%% Clean workspace
clear all; close all; clc

%% loading training set
[images1, labels1] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
%% loading tesing set
[images2, labels2] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
%% projection into PCA space
row_size = size(images1, 1) * size(images1, 2);
image = reshape(images1, row_size, size(images1, 3));
image = im2double(image);
meandata = mean(image, 2);
image = image - repmat(meandata,1,size(images1, 3));
row_size2 = size(images2, 1) * size(images2, 2);
images2 = reshape(images2, row_size2, size(images2, 3));
images2 = im2double(images2);
images2 = images2-repmat(meandata,1,size(images2, 3));
[U,S,V] = svd(image, 'econ');
sing_val = diag(S).^2;
lambda = sing_val.^2;
figure(1)
subplot(1,2,1)
plot(sing_val, 'ko','Linewidth',2)
ylabel('\lambda');
title("Singular Value Spectrum");
subplot(1,2,2)
semilogy(lambda,'ko','Linewidth',2)
ylabel('\lambda (log scale)');
title("Singular Value Spectrum of log scale");

figure(2)
p = U(:,[2,3,5])'*image;
for count = 1:10
    index=find(labels1==count - 1);
    pro=p(:,index);
    plot3(pro(1,:),pro(2,:),pro(3,:),'+');
    hold on
end


%% two digits train
feature = 10;
index1 = find(labels1 == 2)';
index2 = find(labels1 == 3)';
d1set = image(:,index1);
d2set = image(:,index2);
[U,vfirst,vsecond,threshold,w,sortn1,sortn2] = two_trainer(d1set,d2set,index1,index2,feature);
% plot
%plot(vfirst,zeros(size(vfirst,2)),'ob','Linewidth',2)
%hold on
%plot(vsecond,ones(size(vsecond,2)),'dr','Linewidth',2)
%ylim([0 1.2])

subplot(1,2,1)
histogram(sortn1,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-8 4],'Ylim',[0 1000],'Fontsize',14)
title('digit 2')
subplot(1,2,2)
histogram(sortn2,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-8 4],'Ylim',[0 1000],'Fontsize',14)
title('digit 3')
%% two digits train test
indext1 = find(labels1 == 2)';
indext2 = find(labels1 == 3)';
image3 = image(:,[indext1 indext2]);
TestNum = 60000;
TestMat = U'*image3;
pval = w'*TestMat;
ResVec = (pval > threshold);
testLabel = zeros(1,size(indext1,2)+size(indext2,2));
for i = 1:size(indext2,2)
    testLabel(size(indext1,2)+i) = 1;
end
err = abs(ResVec - testLabel);
errNum = sum(err);
sucRate = 1 - errNum/TestNum;
%% two digits test
indext1 = find(labels2 == 2)';
indext2 = find(labels2 == 3)';
image2 = images2(:,[indext1 indext2]);
TestNum = size(image2,2);
TestMat = U'*image2;
pval = w'*TestMat;
ResVec = (pval > threshold);
testLabel = zeros(1,size(indext1,2)+size(indext2,2));
for i = 1:size(indext2,2)
    testLabel(size(indext1,2)+i) = 1;
end
err = abs(ResVec - testLabel);
errNum = sum(err);
sucRate = 1 - errNum/TestNum;
%% finding most/least accurate
feature = 10;
highi1 = 0;
highi2 = 0;
lowi1 = 0;
lowi2 = 0;
highsr = 0;
lowsr = 1;
for i = 1:10
    for j = i+1:10
        index1 = find(labels1 == i-1)';
        index2 = find(labels1 == j-1)';
        d1set = image(:,index1);
        d2set = image(:,index2);
        [U,vfirst,vsecond,threshold,w,sortn1,sortn2] = two_trainer(d1set,d2set,index1,index2,feature);
        indext1 = find(labels2 == i-1)';
        indext2 = find(labels2 == j-1)';
        image2 = images2(:,[indext1 indext2]);
        TestNum = size(image2,2);
        TestMat = U'*image2;
        pval = w'*TestMat;
        ResVec = (pval > threshold);
        testLabel = zeros(1,size(indext1,2)+size(indext2,2));
        for index3 = 1:size(indext2,2)
            testLabel(size(indext1,2)+index3) = 1;
        end
        err = abs(ResVec - testLabel);
        errNum = sum(err);
        sucRate = 1 - errNum/TestNum;
        if sucRate > highsr
            highi1 = i-1;
            highi2 = j-1;
            highsr = sucRate;
        end
        if sucRate < lowsr
            lowi1 = i-1;
            lowi2 = j-1;
            lowsr = sucRate;
        end
    end
end

%% svm
% classification tree on fisheriris data
load fisheriris;
tree = fitctree(image',labels1, 'MaxNumSplits',10,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree, 'mode', 'individual');
[~, k] = min(classError);
testlabels = predict(tree.Trained{k}, images2');
err_tree = immse(testlabels, labels2);
correct = find((testlabels - labels2) == 0);
sRt = size(correct) / size(labels2);
%% three digit
i1 = find(labels1 == 0)';
i2 = find(labels1 == 1)';
i3 = find(labels1 == 2)';
ds1 = image(:,i1);
ds2= image(:,i2);
ds3 = image(:,i3);
[U,v1,v2,v3,t12,t13,t23,w,s1,s2,s3] = three_trainer(ds1,ds2,ds3,i1,i2,i3,20);

subplot(1,2,1)
histogram(s2,30); hold on, plot([t23 t23], [0 1000],'r')
set(gca,'Xlim',[-6 7],'Ylim',[0 1000],'Fontsize',14)
title('digit 1 in 1 and 2')
subplot(1,2,2)
histogram(s3,30); hold on, plot([t13 t13], [0 1000],'r')
set(gca,'Xlim',[-6 7],'Ylim',[0 1000],'Fontsize',14)
title('digit 2 in 1 and 2')
%% functions
function [U,vfirst,vsecond,threshold,w,sortn1,sortn2] = two_trainer(d1set,d2set,index1,index2,feature)
    [U,S,V] = svd([d1set d2set],'econ');
    U = U(:,1:feature);
    digit_proj = S*V';
    d1 = digit_proj(1:feature,1:size(index1,2));
    d2 = digit_proj(1:feature,size(index1,2)+1:size(index1,2)+size(index2,2));

    % scatter matric
    md1 = mean(d1,2);
    md2 = mean(d2,2);

    Sw = 0;
    for k = 1:size(index1,2)
        Sw = Sw+(d1(:,k)-md1)*(d1(:,k)-md1)';
    end
    for k = 1:size(index2,2)
        Sw = Sw+(d2(:,k)-md2)*(d2(:,k)-md2)';
    end
    Sb = (md1-md2)*(md1-md2)';
    
    % find w 
    [V2, D] = eig(Sb, Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    % project onto w
    vfirst = w'*d1;
    vsecond = w'*d2;

    % setup threshold
    if mean(vfirst) > mean(vsecond) 
        w = -w;
        vfirst = -vfirst;
        vsecond = -vsecond;
    end

    sortn1 = sort(vfirst);
    sortn2 = sort(vsecond);
    t1 = length(sortn1);
    t2=1;
    while sortn1(t1) > sortn2(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    threshold = (sortn1(t1) + sortn2(t2))/2;
end

function [U,v1,v2,v3,t12,t13,t23,w,s1,s2,s3] = three_trainer(ds1,ds2,ds3,i1,i2,i3,feature)
    [U,S,V] = svd([ds1 ds2 ds3],'econ');
    U = U(:,1:feature);
    digit_proj = S*V';
    d1 = digit_proj(1:feature,1:size(i1,2));
    d2 = digit_proj(1:feature,size(i1,2)+1:size(i1,2)+size(i2,2));
    d3 = digit_proj(1:feature,size(i1,2)+size(i2,2)+1:size(i1,2)+size(i2,2)+size(i3,2));

    % scatter matric
    md1 = mean(d1,2);
    md2 = mean(d2,2);
    md3 = mean(d3,2);
    mda = [md1 md2 md3];
    
    Sw = 0;
    for k = 1:size(i1,2)
        Sw = Sw+(d1(:,k)-md1)*(d1(:,k)-md1)';
    end
    for k = 1:size(i2,2)
        Sw = Sw+(d2(:,k)-md2)*(d2(:,k)-md2)';
    end
    for k = 1:size(i3,2)
        Sw = Sw+(d3(:,k)-md3)*(d3(:,k)-md3)';
    end
    Sb = (md1-md2)*(md1-md2)';
    meanall = (md1 + md2 + md3) / 3;
    Sb = Sb + (mda(:,1)-meanall)*(mda(:,1)-meanall)'; 
    Sb = Sb + (mda(:,2)-meanall)*(mda(:,2)-meanall)'; 
    Sb = Sb + (mda(:,3)-meanall)*(mda(:,3)-meanall)'; 

    % find w 
    [V2, D] = eig(Sb, Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    % project onto w
    v1 = w'*d1;
    v2 = w'*d2;
    v3 = w'*d3;

    % setup threshold
    if mean(v1) > mean(v2) 
        w = -w;
        v1 = -v1;
        v2 = -v2;
    end
    if mean(v2) > mean(v3) 
        w = -w;
        v2 = -v2;
        v3 = -v3;
    end

    s1 = sort(v1);
    s2 = sort(v2);
    s3 = sort(v3);
    t1 = length(s1);
    t2=1;
    while s1(t1) > s2(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    t12 = (s1(t1) + s2(t2))/2;
    t1 = length(s1);
    t3=1;
    while s1(t1) > s3(t3)
        t1 = t1 - 1;
        t3 = t3 + 1;
    end
    t13 = (s1(t1) + s3(t3))/2;
    t2 = length(s2);
    t3=1;
    while s2(t2) > s3(t3)
        t2 = t2 - 1;
        t3 = t3 + 1;
    end
    t23 = (s2(t2) + s3(t3))/2;
end
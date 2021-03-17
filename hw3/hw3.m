%% load ideal case camN_1
% Clean workspace
clear all; close all; clc
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
%implay(vidFrames1_1)
%implay(vidFrames2_1)
%implay(vidFrames3_1)

%% load noisy case camN_2
% Clean workspace
clear all; close all; clc
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')
%implay(vidFrames1_2)
%implay(vidFrames2_2)
%implay(vidFrames3_2)

%% load Horizontal Displacement case camN_3
% Clean workspace
clear all; close all; clc
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
%implay(vidFrames1_3)
%implay(vidFrames2_3)
%implay(vidFrames3_3)

%% load Horizontal Displacement and Rotation case camN_4
% Clean workspace
clear all; close all; clc
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')
%implay(vidFrames1_4)
%implay(vidFrames2_4)
%implay(vidFrames3_4)

%% constant
Nx = 480;
Ny = 640;
wx = 200;
wy = 200;

%% N = 1 case1

numFrames1 = size(vidFrames1_1,4);
filter = zeros(Nx, Ny);
filter(Nx/4 + wx/4:Nx/2 + wx, Ny/2 - wy/10:Ny/2 + wy/2) = 1;
plotx1 = zeros(1, numFrames1);
ploty1 = zeros(1, numFrames1);

for j = 1:numFrames1
    I = vidFrames1_1(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx1(j) = mean(place1);
    ploty1(j) = mean(place2);
end

%% N = 2 case1
numFrames2 = size(vidFrames2_1,4);
filter = zeros(Nx, Ny);
filter(Nx/4 - wx/10:Nx/2 + wx, Ny/2 - wy/2:Ny/2 + wy/5) = 1;
plotx2 = zeros(1, numFrames2);
ploty2 = zeros(1, numFrames2);

for j = 1:numFrames2
    I = vidFrames2_1(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx2(j) = mean(place1);
    ploty2(j) = mean(place2);
end

%% N = 3 case1
numFrames3 = size(vidFrames3_1,4);
filter = zeros(Nx, Ny);
filter(Nx/2 - wx/10:Nx/2 + wx/2, Ny/4 + wy/2 - wy/10:Ny/2 + wy - wy/10) = 1;
plotx3 = zeros(1, numFrames3);
ploty3 = zeros(1, numFrames3);

for j = 1:numFrames3
    I = vidFrames3_1(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx3(j) = mean(place1);
    ploty3(j) = mean(place2);
end

%% result of first case
minlength = min([numFrames1, numFrames2, numFrames3]);
plotx1 = plotx1(1:minlength);
ploty1 = ploty1(1:minlength);
plotx2 = plotx2(1:minlength);
ploty2 = ploty2(1:minlength);
plotx3 = plotx3(1:minlength);
ploty3 = ploty3(1:minlength);

all = [plotx1; ploty1; plotx2; ploty2; plotx3; ploty3];
center = all - mean(all, 2);

[U,S,V] = svd(center / sqrt(minlength) - 1, 'econ');
disp = U' * center;

figure(1)
X_rank1 = S(1,1)*U(:,1)*V(:,1)';
subplot(4,1,1)
plot(X_rank1(1,:),X_rank1(2,:),'r.','MarkerSize',10);
xlabel('x');
ylabel('y');
title("first rank approximation");
sig = diag(S).^2;
subplot(4,1,2)
plot(sig/sum(sig),'ko','Linewidth',2)
ylabel('\sigma');
xlabel('N');
title("energy");
subplot(4,1,3)
plot(1:minlength, center(1,:), 1:minlength, center(2,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("noise case, displancement along xy plane and z axis");
legend("z", "xy")
subplot(4,1,4)
plot(1:minlength, disp(1,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("noise case");
legend("PCA on the second case")

%% N = 1 case2
numFrames1 = size(vidFrames1_2,4);
filter = zeros(Nx, Ny);
filter(Nx/4 - wx/4:Nx/2 + wx, Ny/2 - wy/10:Ny/2 + wy/2) = 1;
plotx1 = zeros(1, numFrames1);
ploty1 = zeros(1, numFrames1);

for j = 1:numFrames1
    I = vidFrames1_2(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx1(j) = mean(place1);
    ploty1(j) = mean(place2);
end

%% N = 2 case2
numFrames2 = size(vidFrames2_2,4);
filter = zeros(Nx, Ny);
filter(Nx/8 - wx/20:Nx - wx/4, Ny/4 + wy/10:Ny/2 + wy/2) = 1;
plotx2 = zeros(1, numFrames2);
ploty2 = zeros(1, numFrames2);

for j = 1:numFrames2
    I = vidFrames2_2(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx2(j) = mean(place1);
    ploty2(j) = mean(place2);
end

%% N = 3 case2
numFrames3 = size(vidFrames3_2,4);
filter = zeros(Nx, Ny);
filter(Nx/2 - wx/4:Nx/2 + wx/2, Ny/4 + wy/2:Ny/2 + wy/2 + wy/4) = 1;
plotx3 = zeros(1, numFrames3);
ploty3 = zeros(1, numFrames3);

for j = 1:numFrames3
    I = vidFrames3_2(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx3(j) = mean(place1);
    ploty3(j) = mean(place2);
end

%% result of second case
minlength = min([numFrames1, numFrames2, numFrames3]);
plotx1 = plotx1(1:minlength);
ploty1 = ploty1(1:minlength);
plotx2 = plotx2(1:minlength);
ploty2 = ploty2(1:minlength);
plotx3 = plotx3(1:minlength);
ploty3 = ploty3(1:minlength);

all = [plotx1; ploty1; plotx2; ploty2; plotx3; ploty3];
center = all - mean(all, 2);

[U,S,V] = svd(center / sqrt(minlength), 'econ');
disp = U' * center;

figure(1)
X_rank1 = S(1,1)*U(:,1)*V(:,1)';
subplot(4,1,1)
plot(X_rank1(1,:),X_rank1(2,:),'r.','MarkerSize',10);
xlabel('x');
ylabel('y');
title("first rank approximation");
sig = diag(S).^2;
subplot(4,1,2)
plot(sig/sum(sig),'ko','Linewidth',2)
ylabel('\sigma');
xlabel('N');
title("energy");
subplot(4,1,3)
plot(1:minlength, center(1,:), 1:minlength, center(2,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("ideal case, displancement along xy plane and z axis");
legend("z", "xy")
subplot(4,1,4)
plot(1:minlength, disp(1,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("ideal case");
legend("PC1 on the first case")

%% N = 1 case3
numFrames1 = size(vidFrames1_3,4);
filter = zeros(Nx, Ny);
filter(Nx/2:Nx - wx/4, Ny/2 - wy/4:Ny/2 + wy/4) = 1;
plotx1 = zeros(1, numFrames1);
ploty1 = zeros(1, numFrames1);

for j = 1:numFrames1
    I = vidFrames1_3(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx1(j) = mean(place1);
    ploty1(j) = mean(place2);
end

%% N = 2 case3
numFrames2 = size(vidFrames2_3,4);
filter = zeros(Nx, Ny);
filter(Nx/3:Nx - wx/2, Ny/4 + wy/4:Ny/2 + wy/4) = 1;
plotx2 = zeros(1, numFrames2);
ploty2 = zeros(1, numFrames2);

for j = 1:numFrames2
    I = vidFrames2_3(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx2(j) = mean(place1);
    ploty2(j) = mean(place2);
end

%% N = 3 case3
numFrames3 = size(vidFrames3_3,4);
filter = zeros(Nx, Ny);
filter(Nx/2 - wx/4:Nx/2 + wx/2, Ny/4 + wy/4:Ny/2 + wy/2 + wy/4) = 1;
plotx3 = zeros(1, numFrames3);
ploty3 = zeros(1, numFrames3);

for j = 1:numFrames3
    I = vidFrames3_3(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx3(j) = mean(place1);
    ploty3(j) = mean(place2);
end

%% result of third case
minlength = min([numFrames1, numFrames2, numFrames3]);
plotx1 = plotx1(1:minlength);
ploty1 = ploty1(1:minlength);
plotx2 = plotx2(1:minlength);
ploty2 = ploty2(1:minlength);
plotx3 = plotx3(1:minlength);
ploty3 = ploty3(1:minlength);

all = [plotx1; ploty1; plotx2; ploty2; plotx3; ploty3];
center = all - mean(all, 2);

[U,S,V] = svd(center / sqrt(minlength), 'econ');
disp = U' * center;

figure(1)
X_rank1 = S(1,1)*U(:,1)*V(:,1)';
subplot(4,1,1)
plot(X_rank1(1,:),X_rank1(2,:),'r.','MarkerSize',10);
xlabel('x');
ylabel('y');
title("first rank approximation");
sig = diag(S).^2;
subplot(4,1,2)
plot(sig/sum(sig),'ko','Linewidth',2)
ylabel('\sigma');
xlabel('N');
title("energy");
subplot(4,1,3)
plot(1:minlength, center(1,:), 1:minlength, center(2,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("horizontal case, displancement along xy plane and z axis");
legend("z", "xy")
subplot(4,1,4)
plot(1:minlength, disp(1,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("horizontal case");
legend("PC1 on the third case")

%% N = 1 case4
numFrames1 = size(vidFrames1_4,4);
filter = zeros(Nx, Ny);
filter(Nx/2:Nx - wx/5, Ny/2:Ny/2 + wy - wy/4) = 1;
plotx1 = zeros(1, numFrames1);
ploty1 = zeros(1, numFrames1);

for j = 1:numFrames1
    I = vidFrames1_4(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx1(j) = mean(place1);
    ploty1(j) = mean(place2);
end

%% N = 2 case4
numFrames2 = size(vidFrames2_4,4);
filter = zeros(Nx, Ny);
filter(Nx/5:Nx - wx/2, Ny/4 + wy/4:Ny/2 + wy/2) = 1;
plotx2 = zeros(1, numFrames2);
ploty2 = zeros(1, numFrames2);

for j = 1:numFrames2
    I = vidFrames2_4(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 245);
    plotx2(j) = mean(place1);
    ploty2(j) = mean(place2);
end

%% N = 3 case4
numFrames3 = size(vidFrames3_4,4);
filter = zeros(Nx, Ny);
filter(Nx/4:Nx/2 + wx/5, Ny/2:Ny - wy/2) = 1;
plotx3 = zeros(1, numFrames3);
ploty3 = zeros(1, numFrames3);

for j = 1:numFrames3
    I = vidFrames3_4(:,:,:,j);
    I = rgb2gray(I);
    I = im2double(I);
    I = I.*filter;
    I = im2uint8(I); % set to unit8 form
    [place1, place2] = find(I > 225);
    plotx3(j) = mean(place1);
    ploty3(j) = mean(place2);
end

%% result of fourth case
minlength = min([numFrames1, numFrames2, numFrames3]);
plotx1 = plotx1(1:minlength);
ploty1 = ploty1(1:minlength);
plotx2 = plotx2(1:minlength);
ploty2 = ploty2(1:minlength);
plotx3 = plotx3(1:minlength);
ploty3 = ploty3(1:minlength);

all = [plotx1; ploty1; plotx2; ploty2; plotx3; ploty3];
center = all - mean(all, 2);

[U,S,V] = svd(center / sqrt(minlength), 'econ');
disp = U' * center;

figure(1)
X_rank1 = S(1,1)*U(:,1)*V(:,1)';
subplot(4,1,1)
plot(X_rank1(1,:),X_rank1(2,:),'r.','MarkerSize',10);
xlabel('x');
ylabel('y');
title("first rank approximation");
sig = diag(S).^2;
subplot(4,1,2)
plot(sig/sum(sig),'ko','Linewidth',2)
ylabel('\sigma');
xlabel('N');
title("energy");
subplot(4,1,3)
plot(1:minlength, center(1,:), 1:minlength, center(2,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("horizontal case with rotation, displancement along xy plane and z axis");
legend("z", "xy")
subplot(4,1,4)
plot(1:minlength, disp(1,:))
ylabel("Displacement(px)"); xlabel("Time(seconds)"); 
title("horizontal case with rotation");
legend("PC1 on the fourth case")


%% Clean workspace
clear all; close all; clc
%% import data1
v1 = VideoReader('monte_carlo_low.mp4');
%% import data2
v2 = VideoReader('ski_drop_low.mp4');
%% calculation1
dt1 = 1/v1.Framerate;
t1 = 0:dt1:v1.Duration;
vf1 = read(v1);
nf1= get(v1,'NumFrames');
f1 = im2double(vf1(:,:,:,1));
s1 = size(f1,1) * size(f1,2);
a1 = zeros(s1, nf1);

for j = 1:nf1
    f = vf1(:,:,:,j);
    f = rgb2gray(f);
    f = im2double(f);
    a1(:,j) = reshape(f, 540 * 960, []);
    % show the image 
    % imshow(f); drawnow
end
%% calculation2
dt2 = 1/v2.Framerate;
t2 = 0:dt2:v2.Duration;
vf2 = read(v2);
nf2= get(v2,'NumFrames');
f2 = im2double(vf2(:,:,:,1));
s2 = size(f2,1) * size(f2,2);
a2 = zeros(s2, nf2);

for j = 1:nf2
    f = vf2(:,:,:,j);
    f = rgb2gray(f);
    f = im2double(f);
    a2(:,j) = reshape(f, 540 * 960, []);
    % show the image 
    % imshow(f); drawnow
end

%% DMD starts here part1
ax11 = a1(:,1:end-1);
ax12 = a1(:,2:end);
[U1, Sigma1, V1] = svd(ax11,'econ');
plot(diag(Sigma1),'ko','Linewidth',2)
ylabel('\sigmaj')
S1 = U1'*ax12*V1*diag(1./diag(Sigma1));
[eV1, D1] = eig(S1); % compute eigenvalues + eigenvectors
mu1 = diag(D1); % extract eigenvalues
omega1 = log(mu1)/dt1;
Phi1 = U1*eV1;

y10 = Phi1 \ ax11(:,1); % pseudoinverse to get initial conditions
umodes1 = zeros(length(y10), length(t1));
for iter = 1:length(t1)
    umodes1(:,iter) = y10 .* exp(omega1*t1(iter));
end
u_dmd1 = Phi1 * umodes1;

%% DMD starts here part2
ax21 = a2(:,1:end-1);
ax22 = a2(:,2:end);
[U2, Sigma2, V2] = svd(ax21,'econ');
plot(diag(Sigma2),'ko','Linewidth',2)
ylabel('\sigmaj')
S2 = U2'*ax22*V2*diag(1./diag(Sigma2));
[eV2, D2] = eig(S2); % compute eigenvalues + eigenvectors
mu2 = diag(D2); % extract eigenvalues
omega2 = log(mu2)/dt2;
Phi2 = U2*eV2;

y20 = Phi2 \ ax21(:,1); % pseudoinverse to get initial conditions
umodes2 = zeros(length(y20), length(t2));
for iter = 1:length(t2)
    umodes2(:,iter) = y20 .* exp(omega2*t2(iter));
end
u_dmd2 = Phi2 * umodes2;

%% DMD calculation result1
xs1 = ax11 - abs(u_dmd1(:,size(u_dmd1, 2) - 1));
neg1 = xs1 < 0;
R1 = xs1 .* neg1;
un1 = R1 + abs(u_dmd1(:,size(u_dmd1, 2) - 1));
xsn1 = xs1 - R1;
rec1 = xsn1 + un1;
sf11 = size(f1,1);
sf12 = size(f1,2);
show1 = reshape(u_dmd1, [sf11, sf12, length(t1)]);

%% DMD calculation result2
xs2 = ax21 - abs(u_dmd2(:,size(u_dmd2, 2) - 1));
neg2 = xs2 < 0;
R2 = xs2 .* neg2;
un2 = R2 + abs(u_dmd2(:,size(u_dmd2, 2) - 1));
xsn2 = xs2 - R2;
rec2 = xsn2 + un2;
sf21 = size(f2,1);
sf22 = size(f2,2);
show2 = reshape(u_dmd2, [sf21, sf22, length(t2)]);

%% show for 1
for i = 1:nf1
    imshow(im2uint8(show1(:,:,i)))
end
title("Reconstruction sum of X_low_rank + X_Sparse");

%% result
L=40;n=379;
x2 = linspace (0,L,n+1); 
x = x2(1:n);
subplot(2,1,1), waterfall(x,t1,abs(u_dmd1), colormap([0 0 0]))
xlabel('x')
ylabel('t')
zlabel(' | u | ')
title('DMD Solution')
set(gca,'FontSize',16)

%% show for 2
for i = 1:nf2
    imshow(im2uint8(show2(:,:,i)))
end
title("Reconstruction sum of X_low_rank + X_Sparse");

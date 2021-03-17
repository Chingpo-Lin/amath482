% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes

x2 = linspace(-L,L,n+1);
x = x2(1:n); 
y =x; 
z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
ave = zeros(n,n,n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un);
    ave = ave + Unt;
end
ave = abs(fftshift(ave))/49;
[M,I] = max(ave(:));
center = [Kx(I),Ky(I),Kz(I)]

%filter function
tau = 0.5;
filter = exp(-tau*((Kx-center(1)).^2+(Ky-center(2)).^2+(Kz-center(3)).^2));
xp = zeros(1,49);
yp = zeros(1,49);
zp = zeros(1,49);
for k = 1:49
    data = reshape(subdata(:,k),n,n,n);
    ftnd = fftn(data);  
    filt = ftnd.* fftshift(filter);
    unfilt = ifftn(filt);
    unfilt = unfilt / max(unfilt(:));
    
    [M,I] = max(unfilt(:));
    center2 = [X(I),Y(I),Z(I)];
    xp(1,k) = center2(1);
    yp(1,k) = center2(2);
    zp(1,k) = center2(3);
end

plot3(xp, yp, zp, 'b', 'LineWidth', 1)
hold on
plot3(xp(49), yp(49), zp(49), 'ro', 'Markersize', 8)
xlabel('x-direction');
ylabel('y-direction');
zlabel('z-direction');
title("path of submarine");
%q3
position = [xp; yp];

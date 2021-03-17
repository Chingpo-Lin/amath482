
%% GNR
% Clean workspace
clear all; close all; clc

figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
t = (1:length(y)) / Fs;
plot(t,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Sweet Child O'' Mine');
%p8 = audioplayer(y,Fs); playblocking(p8);

n = length(y);
k = (2*pi/tr_gnr)*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

S = y'; % create signal
tau = 0:0.1:tr_gnr;
a = 1300;
Sgt = zeros(length(y), length(tau));
 
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sf = g.*S;
    Sft = fft(Sf);
    [M,I] = max(abs(Sft(:)));
    ftau = -0.2/length(tau);
    filter = Sft.*exp(ftau*((k-k(I)).^2));
    filter(80 > k./(2*pi) | k./(2*pi) > 1200) = 0; % this filter the guitar
    Sgt(:,i) = fftshift(abs(filter));
end

figure(2)
pcolor(tau,ks./(2*pi),Sgt)
shading interp
set(gca,'ylim',[0 1500],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Sweet Child O'' Mine');
%% Floyd
% Clean workspace
clear all; close all; clc

figure(1)
[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
t = (1:length(y)) / Fs;
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Comfortably Numb');
%p8 = audioplayer(y,Fs); playblocking(p8);

n = length(y);
k = (1/tr_gnr)*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

S = y'; % create signal

tau = 0:1:tr_gnr;
a = 5300;
Sgt = zeros(length(y) - 1, length(tau));

for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sf = g.*S;
    Sft = fft(Sf);
    [M,I] = max(abs(Sft(:)));
    s = (k-k(I)).^2;
    ftau = -1/length(tau);
    filter = Sft(1:length(s)).*exp(ftau*((k-k(I)).^2));
    filter(60 > k | k > 250) = 0;
    Sgt(:,i) = fftshift(abs(filter));
end

figure(2)
pcolor(tau,ks,Sgt)
shading interp
set(gca,'ylim',[0 1500],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Comfortably Numb')

%% Floyd Q2
% Clean workspace
clear all; close all; clc

figure(1)
[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
t = (1:length(y)) / Fs;
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Comfortably Numb');
%p8 = audioplayer(y,Fs); playblocking(p8);

n = length(y);
k = (1/tr_gnr)*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

S = y'; % create signal

tau = 0:1:tr_gnr;
a = 5300;
Sgt = zeros(length(y) - 1, length(tau));

for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sf = g.*S;
    Sft = fft(Sf);
    [M,I] = max(abs(Sft(:)));
    s = (k-k(I)).^2;
    ftau = -1/length(tau);
    filter = Sft(1:length(s)).*exp(ftau*((k-k(I)).^2));
    filter(k > 60 & k < 250) = 0; % this filter out the bass
    Sgt(:,i) = fftshift(abs(filter));
end

figure(2)
pcolor(tau,ks,Sgt)
shading interp
set(gca,'ylim',[0 1500],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Comfortably Numb')

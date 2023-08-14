% generateData.m, 产生二阶环节的幅频,相频数据  
clc
clear
wn=5;  % natural frequency
zta=0.2; %damping ratio
num=[wn*wn];  %numerator of 2nd-order system
den=[1,2*zta*wn,wn*wn];  %denominator of 2nd-order system

w=logspace(-1,2);  %10**(-1) ~10**1

[mag,ph,w]=bode(num,den,w); % generate frequency response data

mag=20*log10(mag);

subplot(2,1,1)
semilogx(w,mag,'r')
title('幅频特性')
% xlabel('频率rad/sec')
ylabel('幅度dB')
grid on

%legend('ζ=0.2','ζ=0.4','ζ=0.8');
subplot(2,1,2)
semilogx(w,ph,'b')
title('相频特性')
xlabel('频率rad/sec')
ylabel('相位(度)')
grid on

% format long
% mag_dB=20*log10(mag);
freqResp=[w,mag,ph];  % frequency, magnitude in dB, phase
save freqData.txt freqResp -ascii



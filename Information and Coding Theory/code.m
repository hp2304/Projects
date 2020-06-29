%% Chaotic System
r_vals=[2, 3.2, 3.5, 3.7];

cnt=1;
for r = r_vals
    x = .5*ones(1,2000);
    for i=2:length(x)
       x(i) = (1-x(i-1))*r*x(i-1);
    end
    subplot(2,2,cnt);
    plot(0:length(x)-1, x);
    title(strcat('r = ', num2str(r)));
    ylabel('Output value');
    xlabel('Number of Iterations');
    axis([1000 2000 0 1]);
    cnt = cnt+1;
end

%% Listen to Input audio file
clear;
[input, Fs] = audioread('handel.wav'); 
sound(input, Fs);
waitforbuttonpress;
clear sound
close all;

%% Load the Input audio
clear;
[input, Fs] = audioread('handel.wav', 'native');

% Perform Encryption

password = 'pass';
r1 = 3.81;
r2 = 3.9;

[init_1, init_2] = gen_unique_vals(password);

if init_1 > 1 || init_2 > 1
    disp('Inits greater than 1');
    return;
end

iter1 = init_1;
iter2 = init_2;
for i=1:1000
    iter1 = (1-iter1)*r1*iter1;
    iter2 = (1-iter2)*r2*iter2;
end

chaotic_vals_1 = zeros(1, 256);
chaotic_vals_2 = zeros(1, 256);

for i = 1:256
    iter1 = (1-iter1)*r1*iter1;
    iter2 = (1-iter2)*r2*iter2;
    chaotic_vals_1(i) = iter1;
    chaotic_vals_2(i) = iter2;
end

[~, list1] = sort(chaotic_vals_1, 'ascend');
[~, list2] = sort(chaotic_vals_2, 'ascend');

list1 = list1 - 1;
list2 = list2 - 1;

if max(list1) > 255 || max(list2) > 255
   disp('List values exceeded 255');
   return
end

map = get_map(list1, list2);
encrypted = zeros(size(input));

for i=1:length(encrypted)
    encrypted(i) = map(double(input(i))+32768+1)-32768;
end
encrypted_aud = int16(encrypted);

audiowrite('encrypted.wav', encrypted_aud,Fs);
disp('Encryption successful.')
disp('encrypted.wav has been created in current directory.')

%% Listen to Encrypted audio file
clear;
[encrypted, Fs] = audioread('encrypted.wav'); 
sound(encrypted, Fs);
waitforbuttonpress;
clear sound
close all;


%% Load the Encrypted audio
clear;
[input, Fs] = audioread('encrypted.wav', 'native');

% Perform Decryption

password = 'pass';
r1 = 3.81;
r2 = 3.9;

[init_1, init_2] = gen_unique_vals(password);

if init_1 > 1 || init_2 > 1
    disp('Inits greater than 1');
    return;
end

iter1 = init_1;
iter2 = init_2;
for i=1:1000
    iter1 = (1-iter1)*r1*iter1;
    iter2 = (1-iter2)*r2*iter2;
end

chaotic_vals_1 = zeros(1, 256);
chaotic_vals_2 = zeros(1, 256);

for i = 1:256
    iter1 = (1-iter1)*r1*iter1;
    iter2 = (1-iter2)*r2*iter2;
    chaotic_vals_1(i) = iter1;
    chaotic_vals_2(i) = iter2;
end

[~, list1] = sort(chaotic_vals_1, 'ascend');
[~, list2] = sort(chaotic_vals_2, 'ascend');

list1 = list1 - 1;
list2 = list2 - 1;

if max(list1) > 255 || max(list2) > 255
   disp('List values exceeded 255');
   return
end

map = get_map(list1, list2);
decrypted = zeros(size(input));

for i=1:length(decrypted)
    decrypted(i) = find(map == double(input(i))+32768)-32768-1;
end
decrypted_aud = int16(decrypted);

audiowrite('decrypted.wav', decrypted_aud,Fs);
disp('Decryption successful.')
disp('decrypted.wav has been created in current directory.')

%% Listen to Decrypted audio file
clear;
[decrypted, Fs] = audioread('decrypted.wav'); 
sound(decrypted, Fs);
waitforbuttonpress;
clear sound
close all;



